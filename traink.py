import argparse
import os
from glob import glob
import pandas as pd
from albumentations import Compose, RandomRotate90, Resize, Normalize, Flip
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm

import archs
import losses
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter, str2bool

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image segmentation')
    parser.add_argument('--name', required=True, help='Model name')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='Mini-batch size')
    parser.add_argument('--arch', default='UNext', help='Architecture')
    parser.add_argument('--input_w', type=int, default=256, help='Image width')
    parser.add_argument('--input_h', type=int, default=256, help='Image height')
    parser.add_argument('--dataset', required=True, help='Base directory containing the dataset folds')
    parser.add_argument('--img_ext', default='.png', help='Image file extension')
    parser.add_argument('--mask_ext', default='.png', help='Mask file extension')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--loss', default='BCEDiceLoss', help='Loss function')
    parser.add_argument('--early_stop', type=int, default=20, help='Early stopping patience')

    return parser.parse_args()

def train(config, train_loader, model, criterion, optimizer):
    model.train()
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
    pbar = tqdm(total=len(train_loader), desc="Training")

    for inputs, targets, _ in train_loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        iou, dice = iou_score(outputs, targets)
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), inputs.size(0))
        avg_meters['iou'].update(iou, inputs.size(0))

        pbar.set_postfix(loss=avg_meters['loss'].avg, iou=avg_meters['iou'].avg)
        pbar.update()

    pbar.close()
    return {'loss': avg_meters['loss'].avg, 'iou': avg_meters['iou'].avg}

def validate(config, val_loader, model, criterion):
    model.eval()
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
    pbar = tqdm(total=len(val_loader), desc="Validation")

    with torch.no_grad():
        for inputs, targets, _ in val_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            iou, dice = iou_score(outputs, targets)

            avg_meters['loss'].update(loss.item(), inputs.size(0))
            avg_meters['iou'].update(iou, inputs.size(0))

            pbar.set_postfix(loss=avg_meters['loss'].avg, iou=avg_meters['iou'].avg)
            pbar.update()

    pbar.close()
    return {'loss': avg_meters['loss'].avg, 'iou': avg_meters['iou'].avg}

def main():
    config = vars(parse_args())
    print("Configuration:", config)

    # Data augmentation
    train_transform = Compose([
        RandomRotate90(),
        Flip(),
        Resize(height=config['input_h'], width=config['input_w']),
        Normalize(),
    ])

    val_transform = Compose([
        Resize(height=config['input_h'], width=config['input_w']),
        Normalize(),
    ])

    for fold in range(5):
        print(f"Fold {fold + 1}")
        dataset_dir = os.path.join(config['dataset'], f'data{fold}')

        train_img_paths = glob(os.path.join(dataset_dir, 'train', 'images', '*' + config['img_ext']))
        val_img_paths = glob(os.path.join(dataset_dir, 'val', 'images', '*' + config['img_ext']))

        print(f"Found {len(train_img_paths)} training images in {os.path.join(dataset_dir, 'train', 'images')}")
        print(f"Found {len(val_img_paths)} validation images in {os.path.join(dataset_dir, 'val', 'images')}")

        train_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_img_paths]
        val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_paths]

        if len(train_img_ids) == 0 or len(val_img_ids) == 0:
            print(f"No images found for fold {fold}. Skipping...")
            continue

        # Data loaders
        train_dataset = Dataset(
            img_ids=train_img_ids,
            img_dir=os.path.join(dataset_dir, 'train', 'images'),
            mask_dir=os.path.join(dataset_dir, 'train', 'masks'),
            img_ext=config['img_ext'],
            mask_ext=config['mask_ext'],
            num_classes=1,
            transform=train_transform
        )

        val_dataset = Dataset(
            img_ids=val_img_ids,
            img_dir=os.path.join(dataset_dir, 'val', 'images'),
            mask_dir=os.path.join(dataset_dir, 'val', 'masks'),
            img_ext=config['img_ext'],
            mask_ext=config['mask_ext'],
            num_classes=1,
            transform=val_transform
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=4
        )

        # Model
        model = archs.__dict__[config['arch']](num_classes=1, input_channels=3, deep_supervision=False)
        model.cuda()

        # Loss and optimizer
        criterion = losses.__dict__[config['loss']]().cuda()
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])

        # Logger to track performance
        log = []
        best_loss = float('inf')
        patience = config['early_stop']
        early_stop_counter = 0

        for epoch in range(config['epochs']):
            print(f'Epoch {epoch+1}/{config["epochs"]}')
            train_metrics = train(config, train_loader, model, criterion, optimizer)
            val_metrics = validate(config, val_loader, model, criterion)

            log.append({
                'fold': fold + 1,
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_iou': train_metrics['iou'],
                'val_loss': val_metrics['loss'],
                'val_iou': val_metrics['iou']
            })

            print(f"Epoch {epoch+1} | Train Loss: {train_metrics['loss']:.4f} | Train IoU: {train_metrics['iou']:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val IoU: {val_metrics['iou']:.4f}")

            # Check if we have a new best validation loss
            if val_metrics['loss'] < best_loss:
                best_loss = val_metrics['loss']
                torch.save(model.state_dict(), os.path.join('logs', f'{config["name"]}_fold_{fold}_best.pth'))
                early_stop_counter = 0  # Reset counter if we have a new best
            else:
                early_stop_counter += 1

            # Early stopping
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}. No improvement in validation loss for {patience} epochs.")
                break

        # Save the log for the current fold
        os.makedirs('logs', exist_ok=True)
        log_df = pd.DataFrame(log)
        log_df.to_csv(os.path.join('logs', f'{config["name"]}_fold_{fold}_log.csv'), index=False)

        print(f"Training complete for fold {fold + 1}.")

if __name__ == '__main__':
    main()


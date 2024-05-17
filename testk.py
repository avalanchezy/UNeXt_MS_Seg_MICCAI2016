import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, jaccard_score
from glob import glob
import SimpleITK as sitk
from dataset import Dataset  # Assuming the Dataset class is in a file named dataset.py
import archs  # Assuming the architecture of the model is defined in archs.py
from albumentations import Compose, Resize, Normalize

# Utility functions
def dice_coefficient(pred, target):
    smooth = 1.0  # To avoid division by zero
    intersection = np.sum(pred * target)
    dice = (2.0 * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)
    return dice

def iou_score(pred, target):
    pred = pred.flatten()
    target = target.flatten()
    return jaccard_score(target, pred, average='binary')

# Utility class
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Function to load model
def load_model(model_path, arch, num_classes, input_channels):
    model = archs.__dict__[arch](num_classes=num_classes, input_channels=input_channels, deep_supervision=False)
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()
    return model

# Function to evaluate the model
def evaluate(test_loader, model):
    overall_dice = []
    overall_f1 = []

    patient_dice_scores = {}
    patient_f1_scores = {}

    with torch.no_grad():
        for inputs, targets, info in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            preds = (outputs.sigmoid().cpu().numpy() > 0.5).astype(np.uint8)
            targets = targets.cpu().numpy().astype(np.uint8)

            for i in range(inputs.size(0)):
                img_id = info['img_id'][i]
                patient_id = img_id.split('_')[0]

                pred = preds[i, 0, :, :]
                target = targets[i, 0, :, :]

                dice = dice_coefficient(pred, target)
                f1 = f1_score(target.flatten(), pred.flatten(), average='binary', zero_division=1)

                overall_dice.append(dice)
                overall_f1.append(f1)

                if patient_id not in patient_dice_scores:
                    patient_dice_scores[patient_id] = []
                    patient_f1_scores[patient_id] = []

                patient_dice_scores[patient_id].append(dice)
                patient_f1_scores[patient_id].append(f1)

    average_dice = np.mean(overall_dice)
    average_f1 = np.mean(overall_f1)

    patient_average_dice = {patient_id: np.mean(scores) for patient_id, scores in patient_dice_scores.items()}
    patient_average_f1 = {patient_id: np.mean(scores) for patient_id, scores in patient_f1_scores.items()}

    return average_dice, average_f1, patient_average_dice, patient_average_f1

# Main function
def main():
    # Parameters
    base_dir = 'data'
    arch = 'UNext'  # Model architecture
    num_classes = 1  # Number of output classes
    input_channels = 3  # Number of input channels

    # Transformations for the test set
    test_transform = Compose([
        Resize(height=320, width=256),
        Normalize()
    ])

    overall_dice_all_folds = []
    overall_f1_all_folds = []
    patient_dice_scores_all_folds = {}
    patient_f1_scores_all_folds = {}

    for fold in range(5):
        model_path = f'logs/5fold_fold_{fold}_best.pth'  # Path to the saved model for the fold
        dataset_dir = os.path.join(base_dir, f'data{fold}', 'test')

        # Prepare the test dataset and dataloader
        test_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in glob(os.path.join(dataset_dir, 'images', '*.png'))]

        test_dataset = Dataset(
            img_ids=test_img_ids,
            img_dir=os.path.join(dataset_dir, 'images'),
            mask_dir=os.path.join(dataset_dir, 'masks'),
            img_ext='.png',
            mask_ext='.png',
            num_classes=num_classes,
            transform=test_transform
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # One sample at a time for per-patient analysis
            shuffle=False,
            num_workers=4
        )

        # Load the model
        model = load_model(model_path, arch, num_classes, input_channels)

        # Evaluate the model
        average_dice, average_f1, patient_average_dice, patient_average_f1 = evaluate(test_loader, model)

        overall_dice_all_folds.append(average_dice)
        overall_f1_all_folds.append(average_f1)

        for patient_id in patient_average_dice.keys():
            if patient_id not in patient_dice_scores_all_folds:
                patient_dice_scores_all_folds[patient_id] = []
                patient_f1_scores_all_folds[patient_id] = []
            
            patient_dice_scores_all_folds[patient_id].append(patient_average_dice[patient_id])
            patient_f1_scores_all_folds[patient_id].append(patient_average_f1[patient_id])

        print(f'Fold {fold + 1} - Average Dice: {average_dice:.4f}, Average F1: {average_f1:.4f}')

    # Overall results
    print(f'Overall Average Dice across all folds: {np.mean(overall_dice_all_folds):.4f}')
    print(f'Overall Average F1 across all folds: {np.mean(overall_f1_all_folds):.4f}')

    for patient_id in patient_dice_scores_all_folds.keys():
        print(f'Patient {patient_id} - Overall Average Dice: {np.mean(patient_dice_scores_all_folds[patient_id]):.4f}, Overall Average F1: {np.mean(patient_f1_scores_all_folds[patient_id]):.4f}')

if __name__ == '__main__':
    main()


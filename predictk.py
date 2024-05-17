import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from glob import glob
import SimpleITK as sitk
from dataset import Dataset  # Assuming the Dataset class is in a file named dataset.py
import archs  # Assuming the architecture of the model is defined in archs.py
from albumentations import Compose, Resize, Normalize

# Function to load model
def load_model(model_path, arch, num_classes, input_channels):
    model = archs.__dict__[arch](num_classes=num_classes, input_channels=input_channels, deep_supervision=False)
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()
    return model

# Function to save predictions as images
def save_prediction(pred, output_path):
    pred_image = (pred * 255).astype(np.uint8)
    sitk_img = sitk.GetImageFromArray(pred_image)
    sitk.WriteImage(sitk_img, output_path)

# Main function
def main():
    base_model_path = 'logs/5fold_fold_{fold}_best.pth'  # Path pattern to the saved models
    dataset_base_dir = 'data/data{fold}/test'
    output_base_dir = 'data/data{fold}/test'  # Base directory for predictions
    arch = 'UNext'  # Model architecture
    num_classes = 1  # Number of output classes
    input_channels = 3  # Number of input channels

    # Transformations for the test set
    test_transform = Compose([
        Resize(height=320, width=256),
        Normalize()
    ])

    for fold in range(5):
        model_path = base_model_path.format(fold=fold)
        dataset_dir = dataset_base_dir.format(fold=fold)
        output_dir = output_base_dir.format(fold=fold)

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

        # Generate and save predictions
        with torch.no_grad():
            for inputs, _, info in test_loader:
                inputs = inputs.cuda()
                outputs = model(inputs)
                preds = (outputs.sigmoid().cpu().numpy() > 0.5).astype(np.uint8)

                for i in range(inputs.size(0)):
                    img_id = info['img_id'][i]
                    output_path = os.path.join(output_dir, f'fold{fold}_pred_{img_id}.png')
                    save_prediction(preds[i, 0, :, :], output_path)

        print(f"Predictions for fold {fold} saved successfully.")

if __name__ == '__main__':
    main()


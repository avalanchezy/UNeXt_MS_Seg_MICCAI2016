import SimpleITK as sitk
import numpy as np
import os
import random
from sklearn.model_selection import KFold

def save_image(image, output_path):
    """Save an image to the specified output path with cropping to 320x256 pixels."""
    # Convert SimpleITK image to a NumPy array
    image_np = sitk.GetArrayFromImage(image)
    
    # Crop the image (removing 8 pixels from top and bottom, 3 pixels from left, and 2 pixels from right)
    cropped_image_np = image_np[8:328, 3:259]
    
    # Convert the NumPy array back to a SimpleITK image
    cropped_image_sitk = sitk.GetImageFromArray(cropped_image_np)
    
    # Cast the image to uint8 and rescale intensity
    cropped_image_sitk = sitk.Cast(sitk.RescaleIntensity(cropped_image_sitk), sitk.sitkUInt8)
    
    # Write the image to the specified path
    sitk.WriteImage(cropped_image_sitk, output_path)

def process_case(case_dir, output_dir, case_name):
    """Process each case and save slices to appropriate directories."""
    print(f'Processing case: {case_name}...')

    # Paths to the image files
    mask_path = os.path.join(case_dir, 'Consensus.nii')
    image_path = os.path.join(case_dir, '3DFLAIR.nii')
    
    # Load the images
    mask_image = sitk.ReadImage(mask_path)
    mri_image = sitk.ReadImage(image_path)

    # Assuming that the third dimension Z is the slice direction
    z_slices = range(mask_image.GetSize()[2])
    
    for z_slice in z_slices:
        mask_slice = mask_image[:, :, z_slice]
        image_slice = mri_image[:, :, z_slice]

        mask_output_path = os.path.join(output_dir, 'masks', '0', f"{case_name}_{z_slice:03}.png")
        image_output_path = os.path.join(output_dir, 'images', f"{case_name}_{z_slice:03}.png")
        
        save_image(mask_slice, mask_output_path)
        save_image(image_slice, image_output_path)

    print(f'Finished processing case: {case_name}')

def setup_directories(base_output_dir):
    """Create necessary directories for dataset storage."""
    for subset in ['train', 'val', 'test']:
        subset_dir = os.path.join(base_output_dir, subset)
        os.makedirs(os.path.join(subset_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(subset_dir, 'masks', '0'), exist_ok=True)

# Example usage of directory setup
output_dir = './data'
setup_directories(output_dir)

# Set the root directory of the cases
data_root_dir = '../0_Data_reg_inter_rigid'

# Get a list of all case directories
case_dirs = [d for d in os.listdir(data_root_dir) if os.path.isdir(os.path.join(data_root_dir, d))]

# Shuffle the list of cases
random.shuffle(case_dirs)

# Create a KFold object
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Generate 5-folds
for fold, (train_val_index, test_index) in enumerate(kf.split(case_dirs)):
    fold_dir = os.path.join(output_dir, f'data{fold}')
    setup_directories(fold_dir)

    train_val_cases = [case_dirs[i] for i in train_val_index]
    test_cases = [case_dirs[i] for i in test_index]

    # Further split train_val_cases into training and validation sets
    val_size = max(1, int(0.2 * len(train_val_cases)))  # 20% of the remaining cases for validation
    train_cases = train_val_cases[:-val_size]
    val_cases = train_val_cases[-val_size:]

    # Process training cases
    train_output_dir = os.path.join(fold_dir, 'train')
    for case_name in train_cases:
        case_dir = os.path.join(data_root_dir, case_name)
        process_case(case_dir, train_output_dir, case_name)

    # Process validation cases
    val_output_dir = os.path.join(fold_dir, 'val')
    for case_name in val_cases:
        case_dir = os.path.join(data_root_dir, case_name)
        process_case(case_dir, val_output_dir, case_name)

    # Process test cases
    test_output_dir = os.path.join(fold_dir, 'test')
    for case_name in test_cases:
        case_dir = os.path.join(data_root_dir, case_name)
        process_case(case_dir, test_output_dir, case_name)

    print(f"Fold {fold} created successfully.")


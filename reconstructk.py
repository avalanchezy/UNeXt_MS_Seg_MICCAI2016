import os
import numpy as np
import SimpleITK as sitk
from glob import glob

def load_slices(patient_id, dir_path, file_prefix):
    slice_paths = sorted(glob(os.path.join(dir_path, f'{file_prefix}{patient_id}_*.png')))
    if not slice_paths:
        raise FileNotFoundError(f"No slices found for patient {patient_id} in {dir_path} with prefix {file_prefix}")
    slices = [sitk.GetArrayFromImage(sitk.ReadImage(slice_path)) for slice_path in slice_paths]
    return np.stack(slices, axis=0)

def save_3d_nifti(volume, output_path):
    sitk_img = sitk.GetImageFromArray(volume)
    sitk.WriteImage(sitk_img, output_path)

def main():
    base_dataset_dir = 'data/data{fold}/test'
    base_output_dir = 'data/data{fold}/nii'  # Output directory for 3D NIfTI files

    for fold in range(5):
        dataset_dir = base_dataset_dir.format(fold=fold)
        output_dir = base_output_dir.format(fold=fold)
        os.makedirs(output_dir, exist_ok=True)

        # Extract patient IDs from test image filenames
        test_img_paths = glob(os.path.join(dataset_dir, 'images', '*.png'))
        patient_ids = set([os.path.splitext(os.path.basename(p))[0].split('_')[0] for p in test_img_paths])

        # Process predictions
        for patient_id in patient_ids:
            try:
                print(f'Processing predictions for patient: {patient_id} (fold {fold})...')
                volume = load_slices(patient_id, dataset_dir, file_prefix=f'fold{fold}_pred_')
                output_path = os.path.join(output_dir, f'{patient_id}_fold{fold}_prediction.nii')
                save_3d_nifti(volume, output_path)
            except FileNotFoundError as e:
                print(e)

        print(f"3D NIfTI predictions for fold {fold} created successfully.")

        # Process images
        for patient_id in patient_ids:
            try:
                print(f'Processing images for patient: {patient_id} (fold {fold})...')
                volume = load_slices(patient_id, os.path.join(dataset_dir, 'images'), file_prefix='')
                output_path = os.path.join(output_dir, f'{patient_id}_fold{fold}_image.nii')
                save_3d_nifti(volume, output_path)
            except FileNotFoundError as e:
                print(e)

        print(f"3D NIfTI images for fold {fold} created successfully.")

        # Process masks
        for patient_id in patient_ids:
            try:
                print(f'Processing masks for patient: {patient_id} (fold {fold})...')
                volume = load_slices(patient_id, os.path.join(dataset_dir, 'masks', '0'), file_prefix='')
                output_path = os.path.join(output_dir, f'{patient_id}_fold{fold}_mask.nii')
                save_3d_nifti(volume, output_path)
            except FileNotFoundError as e:
                print(e)

        print(f"3D NIfTI masks for fold {fold} created successfully.")

    print("All 3D NIfTI files created successfully.")

if __name__ == '__main__':
    main()


import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import pandas as pd
import numpy as np
import os

class UterusDataset(Dataset):
    """
    PyTorch Dataset for loading 2D slices from 3D MRI scans of the uterus.
    """
    def __init__(self, manifest_path, transforms=None):
        """
        Args:
            manifest_path (str): Path to the manifest CSV file.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.manifest = pd.read_csv(manifest_path)
        self.transforms = transforms
        self.slice_map = []
        
        print("Loading dataset and creating slice map...")
        for patient_index, row in self.manifest.iterrows():
            mask_image = sitk.ReadImage(row['mask_path'])
            num_slices = mask_image.GetSize()[2]
            
            for slice_index in range(num_slices):
                mask_array = sitk.GetArrayFromImage(mask_image)
                if np.sum(mask_array[slice_index, :, :]) > 0:
                    self.slice_map.append({
                        'patient_index': patient_index,
                        'slice_index': slice_index
                    })
        print(f"Slice map created. Found {len(self.slice_map)} slices containing the uterus.")

    def __len__(self):
        return len(self.slice_map)

    def __getitem__(self, idx):
        slice_info = self.slice_map[idx]
        patient_index = slice_info['patient_index']
        slice_index = slice_info['slice_index']
        
        patient_data = self.manifest.iloc[patient_index]
        mri_path = patient_data['mri_path']
        mask_path = patient_data['mask_path']
        
        mri_image = sitk.ReadImage(mri_path, sitk.sitkFloat32)
        mask_image = sitk.ReadImage(mask_path, sitk.sitkUInt8)
        
        mri_array = sitk.GetArrayFromImage(mri_image)
        mask_array = sitk.GetArrayFromImage(mask_image)
        
        mri_slice = mri_array[slice_index, :, :]
        mask_slice = mask_array[slice_index, :, :]
        
        image_tensor = torch.from_numpy(mri_slice).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_slice).unsqueeze(0).float()
        
        if self.transforms:
            combined = torch.cat([image_tensor, mask_tensor], dim=0)
            combined_transformed = self.transforms(combined)
            image_tensor = combined_transformed[0, :, :].unsqueeze(0)
            mask_tensor = combined_transformed[1, :, :].unsqueeze(0)

        return image_tensor, mask_tensor
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import SimpleITK as sitk
import pandas as pd
import numpy as np
import os

class UterusDataset(Dataset):
    """
    PyTorch Dataset for loading 2D slices from 3D MRI scans of the uterus.
    Handles resizing, normalization, and optional augmentation.
    """
    def __init__(self, manifest_path, image_size=256, augment=False):
        """
        Args:
            manifest_path (str): Path to the manifest CSV file.
            image_size (int): The size to resize images and masks to.
            augment (bool): Whether to apply data augmentation.
        """
        self.manifest = pd.read_csv(manifest_path)
        self.image_size = image_size
        self.augment = augment
        self.slice_map = []
        
        # Define transforms that are always applied
        self.image_transform = T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True)
        self.mask_transform = T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST, antialias=True)
        
        # Define augmentation transforms if requested
        if self.augment:
            self.augmentation_transform = T.Compose([
                T.RandomAffine(degrees=25, translate=(0.1, 0.1)), # Corresponds to paper's +/- 25 degrees and ~25 pixels
                T.RandomHorizontalFlip(p=0.5),
            ])
        
        print(f"Loading manifest from {manifest_path} and creating slice map...")
        # This loop is a bit slow but ensures we only use slices with masks
        for patient_index, row in self.manifest.iterrows():
            mask_image = sitk.ReadImage(row['mask_path'])
            num_slices = mask_image.GetSize()[2]
            mask_array = sitk.GetArrayFromImage(mask_image)
            for slice_index in range(num_slices):
                if np.sum(mask_array[slice_index, :, :]) > 0:
                    self.slice_map.append({'patient_index': patient_index, 'slice_index': slice_index})
        print(f"Slice map created. Found {len(self.slice_map)} slices containing the uterus.")

    def __len__(self):
        return len(self.slice_map)

    def __getitem__(self, idx):
        slice_info = self.slice_map[idx]
        patient_data = self.manifest.iloc[slice_info['patient_index']]
        
        mri_image = sitk.ReadImage(patient_data['mri_path'], sitk.sitkFloat32)
        mask_image = sitk.ReadImage(patient_data['mask_path'], sitk.sitkUInt8)
        
        mri_slice = sitk.GetArrayFromImage(mri_image)[slice_info['slice_index'], :, :]
        mask_slice = sitk.GetArrayFromImage(mask_image)[slice_info['slice_index'], :, :]
        
        image_tensor = torch.from_numpy(mri_slice).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_slice).unsqueeze(0).float()
        
        # Apply resize transforms first
        image_tensor = self.image_transform(image_tensor)
        mask_tensor = self.mask_transform(mask_tensor)
        
        # Apply augmentation if enabled
        if self.augment:
            # Stack image and mask to apply the same random transform
            combined = torch.cat([image_tensor, mask_tensor], dim=0)
            combined = self.augmentation_transform(combined)
            image_tensor, mask_tensor = combined[0].unsqueeze(0), combined[1].unsqueeze(0)
        
        # Normalize the image tensor to [0, 1]
        min_val, max_val = image_tensor.min(), image_tensor.max()
        if max_val > min_val:
            image_tensor = (image_tensor - min_val) / (max_val - min_val)
            
        return image_tensor, mask_tensor
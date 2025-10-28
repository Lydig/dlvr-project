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
    

# --- New Class for Step 2 ---
# We create a new class to handle the paper's specific preprocessing.
class UterusDatasetWithPreprocessing(Dataset):
    """
    PyTorch Dataset for loading 2D slices from 3D MRI scans.
    This version applies the specific preprocessing from the RAovSeg paper.
    """
    def __init__(self, manifest_path, image_size=256, augment=False):
        self.manifest = pd.read_csv(manifest_path)
        self.image_size = image_size
        self.augment = augment
        self.slice_map = []
        
        self.image_transform = T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True)
        self.mask_transform = T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST, antialias=True)
        
        if self.augment:
            self.augmentation_transform = T.Compose([
                T.RandomAffine(degrees=25, translate=(0.1, 0.1)),
                T.RandomHorizontalFlip(p=0.5),
            ])
        
        print(f"Loading manifest from {manifest_path} and creating slice map...")
        for patient_index, row in self.manifest.iterrows():
            # Check if mask_path is valid before processing
            if pd.notna(row['mask_path']) and os.path.exists(row['mask_path']):
                mask_image = sitk.ReadImage(row['mask_path'])
                num_slices = mask_image.GetSize()[2]
                mask_array = sitk.GetArrayFromImage(mask_image)
                for slice_index in range(num_slices):
                    if np.sum(mask_array[slice_index, :, :]) > 0:
                        self.slice_map.append({'patient_index': patient_index, 'slice_index': slice_index})
        print(f"Slice map created. Found {len(self.slice_map)} slices containing the ovary.")

    def __len__(self):
        return len(self.slice_map)

    def _preprocess_raovseg(self, img_np):
        """
        Applies the custom preprocessing from the RAovSeg paper.
        Clips, normalizes, and enhances contrast for ovary regions.
        o1 and o2 are intensity thresholds from the paper.
        """
        o1 = 0.22
        o2 = 0.3
        
        # 1. Percentile clipping
        p1 = np.percentile(img_np, 1)
        p99 = np.percentile(img_np, 99)
        img_clipped = np.clip(img_np, p1, p99)

        # 2. Min-max normalization to [0, 1]
        min_val, max_val = img_clipped.min(), img_clipped.max()
        img_norm = (img_clipped - min_val) / (max_val - min_val) if max_val > min_val else img_clipped

        # 3. Custom intensity mapping from the paper
        img_enhanced = img_norm.copy()
        # Create masks for different conditions
        mask_lt_o1 = img_norm < o1
        mask_gt_o2 = img_norm > o2
        mask_between = (img_norm >= o1) & (img_norm < o2)
        mask_gt_05 = img_norm > 0.5
        
        # Apply transformations based on masks
        img_enhanced[mask_between] = 1.0 # Highlight the ovary range
        img_enhanced[mask_lt_o1] = img_norm[mask_lt_o1] # Keep values below o1
        img_enhanced[mask_gt_o2] = img_norm[mask_gt_o2] # Keep values above o2
        img_enhanced[mask_gt_05] = 1.0 - img_norm[mask_gt_05] # Invert high intensities
        
        return img_enhanced


    def __getitem__(self, idx):
        slice_info = self.slice_map[idx]
        patient_data = self.manifest.iloc[slice_info['patient_index']]
        
        mri_image = sitk.ReadImage(patient_data['mri_path'], sitk.sitkFloat32)
        mask_image = sitk.ReadImage(patient_data['mask_path'], sitk.sitkUInt8)
        
        mri_slice = sitk.GetArrayFromImage(mri_image)[slice_info['slice_index'], :, :]
        
        # --- CHANGE: Apply RAovSeg preprocessing ---
        mri_slice = self._preprocess_raovseg(mri_slice)
        
        mask_slice = sitk.GetArrayFromImage(mask_image)[slice_info['slice_index'], :, :]
        
        image_tensor = torch.from_numpy(mri_slice).unsqueeze(0).float() # Ensure it's float
        mask_tensor = torch.from_numpy(mask_slice).unsqueeze(0).float()
        
        image_tensor = self.image_transform(image_tensor)
        mask_tensor = self.mask_transform(mask_tensor)
        
        if self.augment:
            combined = torch.cat([image_tensor, mask_tensor], dim=0)
            combined = self.augmentation_transform(combined)
            image_tensor, mask_tensor = combined[0].unsqueeze(0), combined[1].unsqueeze(0)
            
        # The normalization part is now inside the _preprocess_raovseg method
        # so we don't need to do it here again.
            
        return image_tensor, mask_tensor
This file is a merged representation of a subset of the codebase, containing files not matching ignore patterns, combined into a single document by Repomix.
The content has been processed where security check has been disabled.

# File Summary

## Purpose
This file contains a packed representation of a subset of the repository's contents that is considered the most important context.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching these patterns are excluded: **/*.pth, **/01_initial_data_exploration.ipynb
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Security check has been disabled - content may contain sensitive information
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
notebooks/
  07_ovary_baseline.ipynb/
    07_ovary_baseline.ipynb
  08_analyze_baseline.ipynb/
    08_analyze_baseline.ipynb
  09_attention_unet.ipynb/
    09_attention_unet.ipynb
  10_analyze_attention_unet.ipynb/
    10_analyze_attention_unet.ipynb
  11_attention_unet_preprocessing.ipynb/
    11_attention_unet_preprocessing.ipynb
  12_analyze_attention_unet_preprocessed.ipynb/
    12_analyze_attention_unet_preprocessed.ipynb
  13_attention_unet_preprocessed_long.ipynb/
    13_attention_unet_preprocessed_long.ipynb
  14_data_deep_dive.ipynb/
    14_data_deep_dive.ipynb
  15_attention_unet_focal_tversky.ipynb/
    15_attention_unet_focal_tversky.ipynb
  16_analyze_with_postprocessing.ipynb/
    16_analyze_with_postprocessing.ipynb
  17_resnet_classifier.ipynb/
    17_resnet_classifier.ipynb
  18_cyclical_lr.ipynb/
    18_cyclical_lr.ipynb
src/
  data_loader.py/
    data_loader.py
  losses.py/
    losses.py
  models.py/
    models.py
  RAovSeg_tools.py/
    RAovSeg_tools.py
```

# Files

## File: notebooks/07_ovary_baseline.ipynb/07_ovary_baseline.ipynb
```
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "6eeeadb8",
   "metadata": {},
   "source": [
    "# Experiment 01: Ovary Segmentation Baseline\n",
    "\n",
    "This notebook establishes a baseline performance for ovary segmentation using a standard U-Net architecture. The goal is to create a reference point before implementing the more advanced techniques from the \"RAovSeg\" paper.\n",
    "\n",
    "### **Model Configuration**\n",
    "\n",
    "*   **Objective**: Re-establish baseline for **Ovary Segmentation**.\n",
    "*   **Model Architecture**: Standard U-Net (from `src/models.py`).\n",
    "*   **Dataset**: D2_TCPW, filtered for patients with both a T2FS MRI and an ovary mask.\n",
    "*   **Input Data**: T2-weighted fat suppression (`_T2FS.nii.gz`) MRI scans.\n",
    "*   **Target Data**: Ovary masks (`_ov.nii.gz`).\n",
    "*   **Preprocessing**: Min-max normalization to [0, 1].\n",
    "*   **Data Augmentation**: Simple `RandomAffine` (rotation, translation) and `RandomHorizontalFlip`.\n",
    "*   **Loss Function**: `DiceBCELoss` (50% Dice, 50% BCE).\n",
    "*   **Optimizer**: Adam.\n",
    "*   **Learning Rate**: 1e-4 (constant).\n",
    "*   **Epochs**: 20.\n",
    "*   **Batch Size**: 1.\n",
    "*   **Image Size**: 256x256.\n",
    "*   **Data Split**: 80% train / 20% validation, split by patient ID."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6faf44f1",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- Main Training Loop (UPDATED) ---\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "print(f\"\\nUsing device: {device}\")\n",
    "\n",
    "baseline_model_ovary = UNet(n_channels=1, n_classes=1).to(device)\n",
    "optimizer = Adam(baseline_model_ovary.parameters(), lr=lr)\n",
    "criterion = DiceBCELoss()\n",
    "\n",
    "train_loss_history, val_loss_history, val_dice_history = [], [], []\n",
    "\n",
    "best_val_dice = -1.0  # Initialize with a value lower than any possible Dice score\n",
    "best_epoch = -1\n",
    "model_save_path = \"../models/07_ovary_baseline_best.pth\"\n",
    "\n",
    "# Ensure the 'models' directory exists\n",
    "os.makedirs(os.path.dirname(model_save_path), exist_ok=True)\n",
    "\n",
    "print(\"\\n--- Starting Ovary Baseline Model Training ---\")\n",
    "for epoch in range(num_epochs):\n",
    "    train_loss = train_one_epoch(baseline_model_ovary, train_loader, optimizer, criterion, device)\n",
    "    val_loss, val_dice = validate(baseline_model_ovary, val_loader, criterion, device)\n",
    "    train_loss_history.append(train_loss)\n",
    "    val_loss_history.append(val_loss)\n",
    "    val_dice_history.append(val_dice)\n",
    "    \n",
    "    print(f\"Epoch {epoch+1}/{num_epochs} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}\")\n",
    "\n",
    "    # Check if this is the best model so far and save it\n",
    "    if val_dice > best_val_dice:\n",
    "        best_val_dice = val_dice\n",
    "        best_epoch = epoch + 1\n",
    "        torch.save(baseline_model_ovary.state_dict(), model_save_path)\n",
    "        print(f\"  -> New best model saved at epoch {best_epoch} with Val Dice: {best_val_dice:.4f}\")\n",
    "\n",
    "print(\"--- Finished Training ---\")\n",
    "print(f\"Best model was from epoch {best_epoch} with a validation Dice score of {best_val_dice:.4f}\")\n",
    "print(f\"Model saved to {model_save_path}\\n\")\n",
    "\n",
    "\n",
    "# --- Visualization ---\n",
    "plt.figure(figsize=(15, 6))\n",
    "plt.subplot(1, 2, 1)\n",
    "plt.plot(range(1, num_epochs + 1), train_loss_history, label='Training Loss', marker='o')\n",
    "plt.plot(range(1, num_epochs + 1), val_loss_history, label='Validation Loss', marker='o')\n",
    "plt.title('Training and Validation Loss (Ovary Baseline)')\n",
    "plt.xlabel('Epochs')\n",
    "plt.ylabel('Loss')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "\n",
    "plt.subplot(1, 2, 2)\n",
    "plt.plot(range(1, num_epochs + 1), val_dice_history, label='Validation Dice Score', color='green', marker='o')\n",
    "plt.title('Validation Dice Score (Ovary Baseline)')\n",
    "plt.xlabel('Epochs')\n",
    "plt.ylabel('Dice Score')\n",
    "# Highlight the best epoch\n",
    "plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Dice @ Epoch {best_epoch}')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "\n",
    "plt.suptitle('Ovary Segmentation Baseline Results', fontsize=16)\n",
    "plt.tight_layout(rect=[0, 0.03, 1, 0.95])\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "dlvr-project",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.23"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

## File: notebooks/08_analyze_baseline.ipynb/08_analyze_baseline.ipynb
```
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "949f1e2a",
   "metadata": {},
   "source": [
    "# Analysis of Ovary Segmentation Baseline (Experiment 01)\n",
    "\n",
    "### Summary\n",
    "This notebook analyzes the performance of our initial baseline model, which was trained to segment ovaries from T2FS MRI scans. The primary goal of this experiment was to pivot from the previous uterus segmentation task to the correct ovary segmentation task and establish a quantitative and qualitative performance baseline. This gives us a benchmark to measure all future improvements against.\n",
    "\n",
    "### Initial Analysis & Course Correction\n",
    "Our first step was a critical review of the project's direction. We identified a major disconnect:\n",
    "1.  **Problem**: The project was initially set up to segment the **uterus**, while the goal is to replicate the \"RAovSeg\" paper, which focuses on **ovary** segmentation.\n",
    "2.  **Investigation**: We attempted to create a new data manifest for ovaries but discovered that the dataset is incomplete—not all patients have ovary masks. This is a common real-world data challenge.\n",
    "3.  **Solution**: We performed a data audit by creating the `d2_data_audit.csv` file. This confirmed that **56 patients** met the paper's criteria of having both a T2FS MRI and a corresponding ovary mask. We then generated a clean manifest, `d2_manifest_t2fs_ovary_eligible.csv`, containing only these eligible patients.\n",
    "\n",
    "### Baseline Methodology\n",
    "The baseline experiment was run in the `07_ovary_baseline.ipynb` notebook with the following setup:\n",
    "*   **Model**: Standard U-Net.\n",
    "*   **Data**: 56 eligible patients from the D2_TCPW dataset, split into 44 for training and 12 for validation.\n",
    "*   **Task**: Segmenting ovaries from T2FS MRI slices.\n",
    "*   **Training**: The model was trained for 20 epochs using the Adam optimizer and a `DiceBCELoss`. We saved the model weights from the epoch with the highest validation Dice score.\n",
    "\n",
    "### Results & Observations\n",
    "1.  **Quantitative Performance**: The baseline performance is poor, confirming that ovary segmentation is a difficult task.\n",
    "    *   The **best validation Dice score achieved was 0.4264 at epoch 3**.\n",
    "    *   The validation loss and Dice curves show that the model begins to **overfit significantly after epoch 3**. The validation metrics become highly unstable while the training loss continues to decrease.\n",
    "\n",
    "2.  **Qualitative Performance (Visual Analysis)**: The sample predictions below confirm the low Dice score.\n",
    "    *   **Poor Localization**: The model correctly identifies the general region of the ovary but fails to produce a precise, well-defined mask.\n",
    "    *   **High False Positives**: The model generates multiple disconnected blobs, indicating it is confused by surrounding tissues and noise. It struggles to distinguish the \"what\" from the \"where\".\n",
    "\n",
    "### Conclusion & Next Steps\n",
    "This baseline experiment was successful in establishing a clear performance benchmark (0.4264 Dice) on the correct task. The key failure mode identified is poor feature localization, leading to imprecise masks and false positives.\n",
    "\n",
    "This provides a strong justification for our next step: implementing an **Attention U-Net**. This architecture is specifically designed to address poor localization by adding attention gates that help the model focus on the most relevant features and ignore noise, which is exactly the problem our baseline model is facing."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d77d770e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- Imports and Setup ---\n",
    "import os\n",
    "import torch\n",
    "import matplotlib.pyplot as plt\n",
    "import sys\n",
    "import numpy as np\n",
    "from torch.utils.data import DataLoader, Subset\n",
    "\n",
    "# Add project root to path\n",
    "project_root = os.path.abspath('..')\n",
    "if project_root not in sys.path:\n",
    "    sys.path.append(project_root)\n",
    "\n",
    "from src.data_loader import UterusDataset\n",
    "from src.models import UNet\n",
    "\n",
    "# --- Configuration ---\n",
    "model_path = \"../models/07_ovary_baseline_best.pth\"\n",
    "manifest_path = '../data/d2_manifest_t2fs_ovary_eligible.csv'\n",
    "image_size = 256\n",
    "num_samples_to_show = 5 # How many examples to visualize\n",
    "\n",
    "# --- Load Model ---\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "model = UNet(n_channels=1, n_classes=1).to(device)\n",
    "\n",
    "try:\n",
    "    model.load_state_dict(torch.load(model_path))\n",
    "    model.eval() # Set model to evaluation mode\n",
    "    print(f\"Successfully loaded model from {model_path}\")\n",
    "except FileNotFoundError:\n",
    "    print(f\"FATAL ERROR: Model file not found at {model_path}. Please make sure you have run the training in notebook 07 and the model was saved.\")\n",
    "    exit()\n",
    "\n",
    "# --- Load Validation Data ---\n",
    "# We only need the validation set to see how the model performs on unseen data.\n",
    "# We create a new dataset instance that does NOT apply augmentations.\n",
    "val_full_dataset = UterusDataset(manifest_path=manifest_path, image_size=image_size, augment=False)\n",
    "\n",
    "# Recreate the same validation split as in the training notebook\n",
    "patient_ids = val_full_dataset.manifest['patient_id'].unique()\n",
    "split_idx = int(len(patient_ids) * 0.8)\n",
    "val_ids = patient_ids[split_idx:]\n",
    "val_indices = [i for i, sm in enumerate(val_full_dataset.slice_map) if val_full_dataset.manifest.loc[sm['patient_index'], 'patient_id'] in val_ids]\n",
    "val_dataset = Subset(val_full_dataset, val_indices)\n",
    "val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False) # Shuffle is False to get consistent results\n",
    "\n",
    "print(f\"Loaded {len(val_dataset)} validation slices.\")\n",
    "\n",
    "# --- Perform and Visualize Inference ---\n",
    "print(f\"\\nVisualizing {num_samples_to_show} sample predictions...\")\n",
    "\n",
    "fig, axes = plt.subplots(num_samples_to_show, 3, figsize=(12, num_samples_to_show * 4))\n",
    "fig.suptitle(\"Baseline Model Predictions on Validation Set\", fontsize=16)\n",
    "\n",
    "with torch.no_grad():\n",
    "    # Loop through the first few samples of the validation loader\n",
    "    for i, (image, mask) in enumerate(val_loader):\n",
    "        if i >= num_samples_to_show:\n",
    "            break\n",
    "            \n",
    "        image = image.to(device)\n",
    "        \n",
    "        # Get model output\n",
    "        output = model(image)\n",
    "        \n",
    "        # Apply sigmoid and threshold to get a binary prediction mask\n",
    "        pred_mask = torch.sigmoid(output) > 0.5\n",
    "        \n",
    "        # Move tensors to CPU and convert to numpy for plotting\n",
    "        image_np = image.cpu().squeeze().numpy()\n",
    "        mask_np = mask.cpu().squeeze().numpy()\n",
    "        pred_mask_np = pred_mask.cpu().squeeze().numpy()\n",
    "        \n",
    "        # Plot Original Image\n",
    "        axes[i, 0].imshow(image_np, cmap='gray')\n",
    "        axes[i, 0].set_title(f\"Sample {i+1}: Input Image\")\n",
    "        axes[i, 0].axis('off')\n",
    "        \n",
    "        # Plot Ground Truth Mask\n",
    "        axes[i, 1].imshow(mask_np, cmap='gray')\n",
    "        axes[i, 1].set_title(\"Ground Truth Mask\")\n",
    "        axes[i, 1].axis('off')\n",
    "        \n",
    "        # Plot Predicted Mask\n",
    "        axes[i, 2].imshow(pred_mask_np, cmap='gray')\n",
    "        axes[i, 2].set_title(\"Predicted Mask\")\n",
    "        axes[i, 2].axis('off')\n",
    "\n",
    "plt.tight_layout(rect=[0, 0.03, 1, 0.96])\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "dlvr-project",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.23"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

## File: notebooks/09_attention_unet.ipynb/09_attention_unet.ipynb
```
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "d458270f",
   "metadata": {},
   "source": [
    "# Experiment 02: Attention U-Net\n",
    "\n",
    "This notebook tests our first major architectural upgrade from the \"RAovSeg\" paper. We are replacing the standard U-Net with an Attention U-Net to see if the attention mechanism can improve performance by helping the model focus on relevant features and suppress noise.\n",
    "\n",
    "### **Model Configuration**\n",
    "\n",
    "*   **Objective**: Evaluate the performance of an **Attention U-Net** on Ovary Segmentation.\n",
    "*   **Model Architecture**: **Attention U-Net (from `src/models.py`)**.\n",
    "*   **Dataset**: D2_TCPW, filtered for patients with both a T2FS MRI and an ovary mask (`d2_manifest_t2fs_ovary_eligible.csv`).\n",
    "*   **Input Data**: T2-weighted fat suppression (`_T2FS.nii.gz`) MRI scans.\n",
    "*   **Target Data**: Ovary masks (`_ov.nii.gz`).\n",
    "*   **Preprocessing**: Min-max normalization to [0, 1].\n",
    "*   **Data Augmentation**: Simple `RandomAffine` (rotation, translation) and `RandomHorizontalFlip`.\n",
    "*   **Loss Function**: `DiceBCELoss` (50% Dice, 50% BCE).\n",
    "*   **Optimizer**: Adam.\n",
    "*   **Learning Rate**: 1e-4 (constant).\n",
    "*   **Epochs**: 20.\n",
    "*   **Batch Size**: 1.\n",
    "*   **Image Size**: 256x256.\n",
    "*   **Data Split**: 80% train / 20% validation, split by patient ID."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8ff0f2bb",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- Imports and Setup ---\n",
    "import os\n",
    "import pandas as pd\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "from torch.utils.data import DataLoader, Subset\n",
    "from torch.optim import Adam\n",
    "from tqdm import tqdm\n",
    "import matplotlib.pyplot as plt\n",
    "import sys\n",
    "\n",
    "# Add project root to path to allow importing from src\n",
    "project_root = os.path.abspath('..')\n",
    "if project_root not in sys.path:\n",
    "    sys.path.append(project_root)\n",
    "\n",
    "from src.data_loader import UterusDataset\n",
    "from src.models import UNet, AttentionUNet # <--- CHANGE: Import AttentionUNet as well\n",
    "\n",
    "# --- Configuration ---\n",
    "manifest_path = '../data/d2_manifest_t2fs_ovary_eligible.csv' # Using the same clean manifest\n",
    "image_size = 256\n",
    "batch_size = 1\n",
    "num_epochs = 20\n",
    "lr = 1e-4\n",
    "\n",
    "# --- Data Loading (Identical to baseline) ---\n",
    "print(\"--- Loading Ovary Data ---\")\n",
    "train_full_dataset = UterusDataset(manifest_path=manifest_path, image_size=image_size, augment=True)\n",
    "val_full_dataset = UterusDataset(manifest_path=manifest_path, image_size=image_size, augment=False)\n",
    "\n",
    "patient_ids = train_full_dataset.manifest['patient_id'].unique()\n",
    "split_idx = int(len(patient_ids) * 0.8)\n",
    "train_ids, val_ids = patient_ids[:split_idx], patient_ids[split_idx:]\n",
    "\n",
    "train_indices = [i for i, sm in enumerate(train_full_dataset.slice_map) if train_full_dataset.manifest.loc[sm['patient_index'], 'patient_id'] in train_ids]\n",
    "val_indices = [i for i, sm in enumerate(val_full_dataset.slice_map) if val_full_dataset.manifest.loc[sm['patient_index'], 'patient_id'] in val_ids]\n",
    "\n",
    "train_dataset = Subset(train_full_dataset, train_indices)\n",
    "val_dataset = Subset(val_full_dataset, val_indices)\n",
    "\n",
    "train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)\n",
    "val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)\n",
    "print(f\"Data successfully split:\\nTraining patients: {len(train_ids)}, Validation patients: {len(val_ids)}\\nTraining slices: {len(train_dataset)}\\nValidation slices: {len(val_dataset)}\")\n",
    "\n",
    "# --- Metrics, Loss, and Training Functions (Identical to baseline) ---\n",
    "def dice_score(preds, targets, epsilon=1e-6):\n",
    "    preds_flat = preds.view(-1)\n",
    "    targets_flat = targets.view(-1)\n",
    "    intersection = (preds_flat * targets_flat).sum()\n",
    "    return (2. * intersection + epsilon) / (preds_flat.sum() + targets_flat.sum() + epsilon)\n",
    "\n",
    "class DiceBCELoss(nn.Module):\n",
    "    def __init__(self, weight=0.5):\n",
    "        super(DiceBCELoss, self).__init__()\n",
    "        self.weight = weight\n",
    "    def forward(self, inputs, targets, smooth=1):\n",
    "        inputs = torch.sigmoid(inputs)\n",
    "        inputs_flat = inputs.view(-1)\n",
    "        targets_flat = targets.view(-1)\n",
    "        bce = F.binary_cross_entropy(inputs_flat, targets_flat, reduction='mean')\n",
    "        intersection = (inputs_flat * targets_flat).sum()\n",
    "        dice_loss = 1 - (2.*intersection + smooth)/(inputs_flat.sum() + targets_flat.sum() + smooth)\n",
    "        return bce * self.weight + dice_loss * (1 - self.weight)\n",
    "\n",
    "def train_one_epoch(model, loader, optimizer, criterion, device):\n",
    "    model.train()\n",
    "    running_loss = 0.0\n",
    "    for images, masks in tqdm(loader, desc=\"Training\"):\n",
    "        images, masks = images.to(device), masks.to(device)\n",
    "        optimizer.zero_grad()\n",
    "        outputs = model(images)\n",
    "        loss = criterion(outputs, masks)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        running_loss += loss.item() * images.size(0)\n",
    "    return running_loss / len(loader.dataset)\n",
    "\n",
    "def validate(model, loader, criterion, device):\n",
    "    model.eval()\n",
    "    running_loss = 0.0\n",
    "    running_dice = 0.0\n",
    "    with torch.no_grad():\n",
    "        for images, masks in tqdm(loader, desc=\"Validation\"):\n",
    "            images, masks = images.to(device), masks.to(device)\n",
    "            outputs = model(images)\n",
    "            loss = criterion(outputs, masks)\n",
    "            preds = torch.sigmoid(outputs) > 0.5\n",
    "            dice = dice_score(preds, masks)\n",
    "            running_loss += loss.item() * images.size(0)\n",
    "            running_dice += dice.item() * images.size(0)\n",
    "    return running_loss / len(loader.dataset), running_dice / len(loader.dataset)\n",
    "\n",
    "# --- Main Training Loop ---\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "print(f\"\\nUsing device: {device}\")\n",
    "\n",
    "# <--- CHANGE: Instantiate the AttentionUNet model ---\n",
    "model = AttentionUNet(n_channels=1, n_classes=1).to(device)\n",
    "optimizer = Adam(model.parameters(), lr=lr)\n",
    "criterion = DiceBCELoss()\n",
    "\n",
    "train_loss_history, val_loss_history, val_dice_history = [], [], []\n",
    "\n",
    "best_val_dice = -1.0\n",
    "best_epoch = -1\n",
    "# <--- CHANGE: New model save path ---\n",
    "model_save_path = \"../models/09_attention_unet_best.pth\"\n",
    "os.makedirs(os.path.dirname(model_save_path), exist_ok=True)\n",
    "\n",
    "print(\"\\n--- Starting Attention U-Net Model Training ---\")\n",
    "for epoch in range(num_epochs):\n",
    "    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)\n",
    "    val_loss, val_dice = validate(model, val_loader, criterion, device)\n",
    "    train_loss_history.append(train_loss)\n",
    "    val_loss_history.append(val_loss)\n",
    "    val_dice_history.append(val_dice)\n",
    "    \n",
    "    print(f\"Epoch {epoch+1}/{num_epochs} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}\")\n",
    "\n",
    "    if val_dice > best_val_dice:\n",
    "        best_val_dice = val_dice\n",
    "        best_epoch = epoch + 1\n",
    "        torch.save(model.state_dict(), model_save_path)\n",
    "        print(f\"  -> New best model saved at epoch {best_epoch} with Val Dice: {best_val_dice:.4f}\")\n",
    "\n",
    "print(\"--- Finished Training ---\")\n",
    "print(f\"Best model was from epoch {best_epoch} with a validation Dice score of {best_val_dice:.4f}\")\n",
    "print(f\"Model saved to {model_save_path}\\n\")\n",
    "\n",
    "# --- Visualization ---\n",
    "plt.figure(figsize=(15, 6))\n",
    "plt.subplot(1, 2, 1)\n",
    "plt.plot(range(1, num_epochs + 1), train_loss_history, label='Training Loss', marker='o')\n",
    "plt.plot(range(1, num_epochs + 1), val_loss_history, label='Validation Loss', marker='o')\n",
    "plt.title('Training and Validation Loss (Attention U-Net)')\n",
    "plt.xlabel('Epochs')\n",
    "plt.ylabel('Loss')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "\n",
    "plt.subplot(1, 2, 2)\n",
    "plt.plot(range(1, num_epochs + 1), val_dice_history, label='Validation Dice Score', color='green', marker='o')\n",
    "plt.title('Validation Dice Score (Attention U-Net)')\n",
    "plt.xlabel('Epochs')\n",
    "plt.ylabel('Dice Score')\n",
    "plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Dice @ Epoch {best_epoch}')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "\n",
    "plt.suptitle('Attention U-Net Results', fontsize=16)\n",
    "plt.tight_layout(rect=[0, 0.03, 1, 0.95])\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "dlvr-project",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.23"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

## File: notebooks/10_analyze_attention_unet.ipynb/10_analyze_attention_unet.ipynb
```
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "cfef0f9b",
   "metadata": {},
   "source": [
    "# Analysis of Attention U-Net (Experiment 02)\n",
    "\n",
    "### Summary\n",
    "This notebook analyzes the performance of the Attention U-Net model from `09_attention_unet.ipynb`. We compare its results directly to the baseline U-Net from Experiment 01 to isolate the impact of the attention mechanism.\n",
    "\n",
    "### Initial Findings\n",
    "- **Quantitative Performance**: The Attention U-Net achieved a best validation Dice score of **0.3920**, which is **lower** than the baseline's 0.4264.\n",
    "- **Instability & Overfitting**: The model's performance is highly unstable, and it begins to overfit very early (peaking at epoch 2).\n",
    "\n",
    "**Hypothesis**: The Attention U-Net is a more complex model. Without specialized preprocessing or a loss function tailored to the task, its increased capacity makes it more prone to overfitting on our dataset, leading to unstable and slightly worse results than the simpler baseline.\n",
    "\n",
    "### Qualitative Performance (Visual Analysis)\n",
    "The sample predictions confirm the low Dice score and support our hypothesis.\n",
    "- **Failure Mode**: The model consistently identifies multiple high-intensity regions as potential ovaries, resulting in a high number of false positives.\n",
    "- **Insight**: The attention mechanism alone is insufficient. The raw image data does not provide a clear enough signal for the model to distinguish the true ovary from other visually similar tissues (e.g., cysts, fluid, parts of the bowel). This strongly motivates the need for a preprocessing step that enhances the contrast of the target organ.\n",
    "\n",
    "### Conclusion & Next Steps\n",
    "This experiment demonstrates that a more complex architecture does not guarantee better performance, especially when the underlying data is challenging. The model's failure to focus correctly points directly to the next logical step in replicating the \"RAovSeg\" paper: implementing their **custom preprocessing function**. This function is specifically designed to make the ovaries \"pop\" in the image, which should provide a much stronger signal for the Attention U-Net to lock onto."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f325d24c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- Imports and Setup ---\n",
    "import os\n",
    "import torch\n",
    "import matplotlib.pyplot as plt\n",
    "import sys\n",
    "import numpy as np\n",
    "from torch.utils.data import DataLoader, Subset\n",
    "\n",
    "# Add project root to path\n",
    "project_root = os.path.abspath('..')\n",
    "if project_root not in sys.path:\n",
    "    sys.path.append(project_root)\n",
    "\n",
    "from src.data_loader import UterusDataset\n",
    "from src.models import AttentionUNet # <--- CHANGE: We are now loading the AttentionUNet\n",
    "\n",
    "# --- Configuration ---\n",
    "# <--- CHANGE: Path to the new best model ---\n",
    "model_path = \"../models/09_attention_unet_best.pth\" \n",
    "manifest_path = '../data/d2_manifest_t2fs_ovary_eligible.csv'\n",
    "image_size = 256\n",
    "num_samples_to_show = 5\n",
    "\n",
    "# --- Load Model ---\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "model = AttentionUNet(n_channels=1, n_classes=1).to(device) # <--- CHANGE: Instantiate AttentionUNet\n",
    "\n",
    "try:\n",
    "    model.load_state_dict(torch.load(model_path))\n",
    "    model.eval()\n",
    "    print(f\"Successfully loaded model from {model_path}\")\n",
    "except FileNotFoundError:\n",
    "    print(f\"FATAL ERROR: Model file not found at {model_path}. Please run the training in notebook 09.\")\n",
    "    exit()\n",
    "\n",
    "# --- Load Validation Data (Identical to before) ---\n",
    "val_full_dataset = UterusDataset(manifest_path=manifest_path, image_size=image_size, augment=False)\n",
    "patient_ids = val_full_dataset.manifest['patient_id'].unique()\n",
    "split_idx = int(len(patient_ids) * 0.8)\n",
    "val_ids = patient_ids[split_idx:]\n",
    "val_indices = [i for i, sm in enumerate(val_full_dataset.slice_map) if val_full_dataset.manifest.loc[sm['patient_index'], 'patient_id'] in val_ids]\n",
    "val_dataset = Subset(val_full_dataset, val_indices)\n",
    "val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)\n",
    "\n",
    "print(f\"Loaded {len(val_dataset)} validation slices.\")\n",
    "\n",
    "# --- Perform and Visualize Inference ---\n",
    "print(f\"\\nVisualizing {num_samples_to_show} sample predictions...\")\n",
    "\n",
    "fig, axes = plt.subplots(num_samples_to_show, 3, figsize=(12, num_samples_to_show * 4))\n",
    "fig.suptitle(\"Attention U-Net Predictions on Validation Set\", fontsize=16)\n",
    "\n",
    "with torch.no_grad():\n",
    "    for i, (image, mask) in enumerate(val_loader):\n",
    "        if i >= num_samples_to_show:\n",
    "            break\n",
    "        image = image.to(device)\n",
    "        output = model(image)\n",
    "        pred_mask = torch.sigmoid(output) > 0.5\n",
    "        \n",
    "        image_np = image.cpu().squeeze().numpy()\n",
    "        mask_np = mask.cpu().squeeze().numpy()\n",
    "        pred_mask_np = pred_mask.cpu().squeeze().numpy()\n",
    "        \n",
    "        axes[i, 0].imshow(image_np, cmap='gray')\n",
    "        axes[i, 0].set_title(f\"Sample {i+1}: Input Image\")\n",
    "        axes[i, 0].axis('off')\n",
    "        \n",
    "        axes[i, 1].imshow(mask_np, cmap='gray')\n",
    "        axes[i, 1].set_title(\"Ground Truth Mask\")\n",
    "        axes[i, 1].axis('off')\n",
    "        \n",
    "        axes[i, 2].imshow(pred_mask_np, cmap='gray')\n",
    "        axes[i, 2].set_title(\"Predicted Mask\")\n",
    "        axes[i, 2].axis('off')\n",
    "\n",
    "plt.tight_layout(rect=[0, 0.03, 1, 0.96])\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "dlvr-project",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.23"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

## File: notebooks/11_attention_unet_preprocessing.ipynb/11_attention_unet_preprocessing.ipynb
```
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "4cb3de3d",
   "metadata": {},
   "source": [
    "# Experiment 03: Attention U-Net with Custom Preprocessing\n",
    "\n",
    "This notebook tests the impact of the custom preprocessing function described in the \"RAovSeg\" paper. We will use the same Attention U-Net architecture as the previous experiment, but we will now feed it images that have been enhanced by the specialized preprocessing logic.\n",
    "\n",
    "### **Model Configuration**\n",
    "\n",
    "*   **Objective**: Evaluate if custom preprocessing improves Attention U-Net performance.\n",
    "*   **Model Architecture**: Attention U-Net (from `src/models.py`).\n",
    "*   **Dataset**: D2_TCPW, eligible patients (`d2_manifest_t2fs_ovary_eligible.csv`).\n",
    "*   **Preprocessing**: **RAovSeg custom preprocessing function applied** (from the new `UterusDatasetWithPreprocessing` class).\n",
    "*   **Data Augmentation**: Simple `RandomAffine` and `RandomHorizontalFlip`.\n",
    "*   **Loss Function**: `DiceBCELoss`.\n",
    "*   **Optimizer**: Adam.\n",
    "*   **Learning Rate**: 1e-4 (constant).\n",
    "*   **Epochs**: 20.\n",
    "*   **Batch Size**: 1.\n",
    "*   **Image Size**: 256x256.\n",
    "*   **Data Split**: 80% train / 20% validation, split by patient ID."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c1bf3651",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- Sanity Check: Visualize Preprocessing ---\n",
    "import SimpleITK as sitk\n",
    "import numpy as np\n",
    "\n",
    "# We'll use the same dataset object to access its methods and manifest\n",
    "# augment=False so we get a clean comparison without random flipping/rotation\n",
    "print(\"--- Running Preprocessing Sanity Check ---\")\n",
    "dataset_for_viz = UterusDatasetWithPreprocessing(manifest_path=manifest_path, image_size=image_size, augment=False)\n",
    "\n",
    "# Pick a sample slice to inspect\n",
    "sample_idx = 30 # You can change this to see different slices\n",
    "slice_info = dataset_for_viz.slice_map[sample_idx]\n",
    "patient_data = dataset_for_viz.manifest.iloc[slice_info['patient_index']]\n",
    "\n",
    "# --- Load the ORIGINAL slice manually ---\n",
    "mri_image_original = sitk.ReadImage(patient_data['mri_path'], sitk.sitkFloat32)\n",
    "mri_slice_original_np = sitk.GetArrayFromImage(mri_image_original)[slice_info['slice_index'], :, :]\n",
    "\n",
    "# --- Get the PREPROCESSED slice using our new dataset class method ---\n",
    "# The __getitem__ method will automatically call our _preprocess_raovseg function\n",
    "preprocessed_tensor, _ = dataset_for_viz[sample_idx]\n",
    "preprocessed_np = preprocessed_tensor.cpu().squeeze().numpy()\n",
    "\n",
    "\n",
    "# --- Plot for comparison ---\n",
    "fig, axes = plt.subplots(1, 2, figsize=(10, 5))\n",
    "fig.suptitle(\"Preprocessing Sanity Check\", fontsize=16)\n",
    "\n",
    "# Plot Original (after simple min-max normalization for visualization)\n",
    "min_val, max_val = mri_slice_original_np.min(), mri_slice_original_np.max()\n",
    "original_normalized = (mri_slice_original_np - min_val) / (max_val - min_val) if max_val > min_val else mri_slice_original_np\n",
    "axes[0].imshow(original_normalized, cmap='gray')\n",
    "axes[0].set_title(\"Original Slice (Normalized)\")\n",
    "axes[0].axis('off')\n",
    "\n",
    "# Plot Preprocessed\n",
    "axes[1].imshow(preprocessed_np, cmap='gray')\n",
    "axes[1].set_title(\"After RAovSeg Preprocessing\")\n",
    "axes[1].axis('off')\n",
    "\n",
    "plt.tight_layout(rect=[0, 0.03, 1, 0.95])\n",
    "plt.show()\n",
    "print(\"--- Sanity Check Complete ---\\n\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b52678d7",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- Imports and Setup ---\n",
    "import os\n",
    "import pandas as pd\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "from torch.utils.data import DataLoader, Subset\n",
    "from torch.optim import Adam\n",
    "from tqdm import tqdm\n",
    "import matplotlib.pyplot as plt\n",
    "import sys\n",
    "\n",
    "# Add project root to path\n",
    "project_root = os.path.abspath('..')\n",
    "if project_root not in sys.path:\n",
    "    sys.path.append(project_root)\n",
    "\n",
    "# <--- CHANGE: Import our new Dataset class ---\n",
    "from src.data_loader import UterusDatasetWithPreprocessing \n",
    "from src.models import AttentionUNet\n",
    "\n",
    "# --- Configuration ---\n",
    "manifest_path = '../data/d2_manifest_t2fs_ovary_eligible.csv'\n",
    "image_size = 256\n",
    "batch_size = 1\n",
    "num_epochs = 20\n",
    "lr = 1e-4\n",
    "\n",
    "# --- Data Loading ---\n",
    "print(\"--- Loading Ovary Data with Custom Preprocessing ---\")\n",
    "# <--- CHANGE: Use the new Dataset class ---\n",
    "train_full_dataset = UterusDatasetWithPreprocessing(manifest_path=manifest_path, image_size=image_size, augment=True)\n",
    "val_full_dataset = UterusDatasetWithPreprocessing(manifest_path=manifest_path, image_size=image_size, augment=False)\n",
    "\n",
    "# Data splitting logic remains the same\n",
    "patient_ids = train_full_dataset.manifest['patient_id'].unique()\n",
    "split_idx = int(len(patient_ids) * 0.8)\n",
    "train_ids, val_ids = patient_ids[:split_idx], patient_ids[split_idx:]\n",
    "train_indices = [i for i, sm in enumerate(train_full_dataset.slice_map) if train_full_dataset.manifest.loc[sm['patient_index'], 'patient_id'] in train_ids]\n",
    "val_indices = [i for i, sm in enumerate(val_full_dataset.slice_map) if val_full_dataset.manifest.loc[sm['patient_index'], 'patient_id'] in val_ids]\n",
    "train_dataset = Subset(train_full_dataset, train_indices)\n",
    "val_dataset = Subset(val_full_dataset, val_indices)\n",
    "train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)\n",
    "val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)\n",
    "print(f\"Data successfully split:\\nTraining patients: {len(train_ids)}, Validation patients: {len(val_ids)}\\nTraining slices: {len(train_dataset)}\\nValidation slices: {len(val_dataset)}\")\n",
    "\n",
    "\n",
    "# --- Metrics, Loss, and Training Functions (Identical to previous notebooks) ---\n",
    "def dice_score(preds, targets, epsilon=1e-6):\n",
    "    preds_flat = preds.view(-1)\n",
    "    targets_flat = targets.view(-1)\n",
    "    intersection = (preds_flat * targets_flat).sum()\n",
    "    return (2. * intersection + epsilon) / (preds_flat.sum() + targets_flat.sum() + epsilon)\n",
    "\n",
    "class DiceBCELoss(nn.Module):\n",
    "    def __init__(self, weight=0.5):\n",
    "        super(DiceBCELoss, self).__init__()\n",
    "        self.weight = weight\n",
    "    def forward(self, inputs, targets, smooth=1):\n",
    "        inputs = torch.sigmoid(inputs)\n",
    "        inputs_flat = inputs.view(-1)\n",
    "        targets_flat = targets.view(-1)\n",
    "        bce = F.binary_cross_entropy(inputs_flat, targets_flat, reduction='mean')\n",
    "        intersection = (inputs_flat * targets_flat).sum()\n",
    "        dice_loss = 1 - (2.*intersection + smooth)/(inputs_flat.sum() + targets_flat.sum() + smooth)\n",
    "        return bce * self.weight + dice_loss * (1 - self.weight)\n",
    "\n",
    "def train_one_epoch(model, loader, optimizer, criterion, device):\n",
    "    model.train()\n",
    "    running_loss = 0.0\n",
    "    for images, masks in tqdm(loader, desc=\"Training\"):\n",
    "        images, masks = images.to(device), masks.to(device)\n",
    "        optimizer.zero_grad()\n",
    "        outputs = model(images)\n",
    "        loss = criterion(outputs, masks)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        running_loss += loss.item() * images.size(0)\n",
    "    return running_loss / len(loader.dataset)\n",
    "\n",
    "def validate(model, loader, criterion, device):\n",
    "    model.eval()\n",
    "    running_loss = 0.0\n",
    "    running_dice = 0.0\n",
    "    with torch.no_grad():\n",
    "        for images, masks in tqdm(loader, desc=\"Validation\"):\n",
    "            images, masks = images.to(device), masks.to(device)\n",
    "            outputs = model(images)\n",
    "            loss = criterion(outputs, masks)\n",
    "            preds = torch.sigmoid(outputs) > 0.5\n",
    "            dice = dice_score(preds, masks)\n",
    "            running_loss += loss.item() * images.size(0)\n",
    "            running_dice += dice.item() * images.size(0)\n",
    "    return running_loss / len(loader.dataset), running_dice / len(loader.dataset)\n",
    "\n",
    "\n",
    "# --- Main Training Loop ---\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "print(f\"\\nUsing device: {device}\")\n",
    "\n",
    "model = AttentionUNet(n_channels=1, n_classes=1).to(device)\n",
    "optimizer = Adam(model.parameters(), lr=lr)\n",
    "criterion = DiceBCELoss()\n",
    "\n",
    "train_loss_history, val_loss_history, val_dice_history = [], [], []\n",
    "\n",
    "best_val_dice = -1.0\n",
    "best_epoch = -1\n",
    "# <--- CHANGE: New model save path ---\n",
    "model_save_path = \"../models/11_attention_unet_preprocessed_best.pth\"\n",
    "os.makedirs(os.path.dirname(model_save_path), exist_ok=True)\n",
    "\n",
    "print(\"\\n--- Starting Attention U-Net with Preprocessing ---\")\n",
    "for epoch in range(num_epochs):\n",
    "    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)\n",
    "    val_loss, val_dice = validate(model, val_loader, criterion, device)\n",
    "    train_loss_history.append(train_loss)\n",
    "    val_loss_history.append(val_loss)\n",
    "    val_dice_history.append(val_dice)\n",
    "    \n",
    "    print(f\"Epoch {epoch+1}/{num_epochs} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}\")\n",
    "\n",
    "    if val_dice > best_val_dice:\n",
    "        best_val_dice = val_dice\n",
    "        best_epoch = epoch + 1\n",
    "        torch.save(model.state_dict(), model_save_path)\n",
    "        print(f\"  -> New best model saved at epoch {best_epoch} with Val Dice: {best_val_dice:.4f}\")\n",
    "\n",
    "print(\"--- Finished Training ---\")\n",
    "print(f\"Best model was from epoch {best_epoch} with a validation Dice score of {best_val_dice:.4f}\")\n",
    "print(f\"Model saved to {model_save_path}\\n\")\n",
    "\n",
    "# --- Visualization ---\n",
    "plt.figure(figsize=(15, 6))\n",
    "plt.subplot(1, 2, 1)\n",
    "plt.plot(range(1, num_epochs + 1), train_loss_history, label='Training Loss', marker='o')\n",
    "plt.plot(range(1, num_epochs + 1), val_loss_history, label='Validation Loss', marker='o')\n",
    "plt.title('Training and Validation Loss (Attn U-Net + Preprocessing)')\n",
    "plt.xlabel('Epochs')\n",
    "plt.ylabel('Loss')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "\n",
    "plt.subplot(1, 2, 2)\n",
    "plt.plot(range(1, num_epochs + 1), val_dice_history, label='Validation Dice Score', color='green', marker='o')\n",
    "plt.title('Validation Dice Score (Attn U-Net + Preprocessing)')\n",
    "plt.xlabel('Epochs')\n",
    "plt.ylabel('Dice Score')\n",
    "plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Dice @ Epoch {best_epoch}')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "\n",
    "plt.suptitle('Attention U-Net w/ Custom Preprocessing Results', fontsize=16)\n",
    "plt.tight_layout(rect=[0, 0.03, 1, 0.95])\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "dlvr-project",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.23"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

## File: notebooks/12_analyze_attention_unet_preprocessed.ipynb/12_analyze_attention_unet_preprocessed.ipynb
```
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "863682d8",
   "metadata": {},
   "source": [
    "# Analysis of Attention U-Net with Custom Preprocessing (Experiment 03)\n",
    "\n",
    "### Summary\n",
    "This notebook analyzes the performance of the Attention U-Net model trained for 20 epochs with the RAovSeg custom preprocessing function. We compare its results to the previous experiments to understand the impact of this preprocessing step.\n",
    "\n",
    "### Quantitative Findings\n",
    "- **Performance**: The model achieved its best validation Dice score of **0.3104 at epoch 20**. This is lower than the baseline's peak score (0.4264).\n",
    "- **Training Dynamics**: Crucially, the validation loss was still generally decreasing and the validation Dice score was still on an upward trend at the end of training. This indicates that the model **did not overfit** within 20 epochs and was likely **stopped too early**.\n",
    "\n",
    "### Qualitative Findings (Visual Analysis)\n",
    "The sample predictions reveal a significant qualitative improvement over the previous models:\n",
    "- **Reduced False Positives**: The model produces much cleaner predictions with far fewer scattered, incorrect blobs. In `Sample 1`, it correctly predicts nothing, whereas previous models produced multiple false positives.\n",
    "- **Improved Localization Focus**: The model is no longer \"guessing\" at every bright spot. It is learning to identify a smaller, more specific set of candidate regions.\n",
    "\n",
    "### Conclusion & Next Steps\n",
    "The custom preprocessing is successful at regularizing the model and preventing the rapid overfitting we saw previously. While the Dice score is lower after 20 epochs, the qualitative results are much better, and the training curves indicate that the model simply needs more time to learn.\n",
    "\n",
    "The clear next step is to **train this same model for more epochs** to allow it to converge and find its true peak performance."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "950c6414",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- Imports and Setup ---\n",
    "import os\n",
    "import torch\n",
    "import matplotlib.pyplot as plt\n",
    "import sys\n",
    "import numpy as np\n",
    "from torch.utils.data import DataLoader, Subset\n",
    "\n",
    "# Add project root to path\n",
    "project_root = os.path.abspath('..')\n",
    "if project_root not in sys.path:\n",
    "    sys.path.append(project_root)\n",
    "\n",
    "# <--- Use the NEW dataset class that has the preprocessing logic ---\n",
    "from src.data_loader import UterusDatasetWithPreprocessing \n",
    "from src.models import AttentionUNet\n",
    "\n",
    "# --- Configuration ---\n",
    "# <--- Path to the new best model from notebook 11 ---\n",
    "model_path = \"../models/11_attention_unet_preprocessed_best.pth\" \n",
    "manifest_path = '../data/d2_manifest_t2fs_ovary_eligible.csv'\n",
    "image_size = 256\n",
    "num_samples_to_show = 5\n",
    "\n",
    "# --- Load Model ---\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "model = AttentionUNet(n_channels=1, n_classes=1).to(device)\n",
    "\n",
    "try:\n",
    "    model.load_state_dict(torch.load(model_path))\n",
    "    model.eval()\n",
    "    print(f\"Successfully loaded model from {model_path}\")\n",
    "except FileNotFoundError:\n",
    "    print(f\"FATAL ERROR: Model file not found at {model_path}. Please run the training in notebook 11.\")\n",
    "    exit()\n",
    "\n",
    "# --- Load Validation Data with Preprocessing ---\n",
    "# We MUST use the same preprocessing on the validation data for analysis\n",
    "val_full_dataset = UterusDatasetWithPreprocessing(manifest_path=manifest_path, image_size=image_size, augment=False)\n",
    "\n",
    "patient_ids = val_full_dataset.manifest['patient_id'].unique()\n",
    "split_idx = int(len(patient_ids) * 0.8)\n",
    "val_ids = patient_ids[split_idx:]\n",
    "val_indices = [i for i, sm in enumerate(val_full_dataset.slice_map) if val_full_dataset.manifest.loc[sm['patient_index'], 'patient_id'] in val_ids]\n",
    "val_dataset = Subset(val_full_dataset, val_indices)\n",
    "val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)\n",
    "\n",
    "print(f\"Loaded {len(val_dataset)} validation slices.\")\n",
    "\n",
    "# --- Perform and Visualize Inference ---\n",
    "print(f\"\\nVisualizing {num_samples_to_show} sample predictions...\")\n",
    "\n",
    "# Note: The \"Input Image\" shown here will be AFTER preprocessing, which is what the model actually sees.\n",
    "fig, axes = plt.subplots(num_samples_to_show, 3, figsize=(12, num_samples_to_show * 4))\n",
    "fig.suptitle(\"Preprocessed Attention U-Net Predictions\", fontsize=16)\n",
    "\n",
    "with torch.no_grad():\n",
    "    for i, (image, mask) in enumerate(val_loader):\n",
    "        if i >= num_samples_to_show:\n",
    "            break\n",
    "        image, mask = image.to(device), mask.to(device)\n",
    "        output = model(image)\n",
    "        pred_mask = torch.sigmoid(output) > 0.5\n",
    "        \n",
    "        image_np = image.cpu().squeeze().numpy()\n",
    "        mask_np = mask.cpu().squeeze().numpy()\n",
    "        pred_mask_np = pred_mask.cpu().squeeze().numpy()\n",
    "        \n",
    "        axes[i, 0].imshow(image_np, cmap='gray')\n",
    "        axes[i, 0].set_title(f\"Sample {i+1}: Preprocessed Input\")\n",
    "        axes[i, 0].axis('off')\n",
    "        \n",
    "        axes[i, 1].imshow(mask_np, cmap='gray')\n",
    "        axes[i, 1].set_title(\"Ground Truth Mask\")\n",
    "        axes[i, 1].axis('off')\n",
    "        \n",
    "        axes[i, 2].imshow(pred_mask_np, cmap='gray')\n",
    "        axes[i, 2].set_title(\"Predicted Mask\")\n",
    "        axes[i, 2].axis('off')\n",
    "\n",
    "plt.tight_layout(rect=[0, 0.03, 1, 0.96])\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "dlvr-project",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.23"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

## File: notebooks/13_attention_unet_preprocessed_long.ipynb/13_attention_unet_preprocessed_long.ipynb
```
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "f001e961",
   "metadata": {},
   "source": [
    "# Experiment 04: Attention U-Net w/ Preprocessing (Longer Training)\n",
    "\n",
    "This notebook is a continuation of Experiment 03. We are re-running the same setup (Attention U-Net + Custom Preprocessing) but for a longer duration to allow the model to fully converge.\n",
    "\n",
    "### **Model Configuration**\n",
    "\n",
    "*   **Objective**: Find the peak performance of the Attention U-Net with custom preprocessing by training for more epochs.\n",
    "*   **Model Architecture**: Attention U-Net.\n",
    "*   **Dataset**: D2_TCPW, eligible patients (`d2_manifest_t2fs_ovary_eligible.csv`).\n",
    "*   **Preprocessing**: RAovSeg custom preprocessing.\n",
    "*   **Data Augmentation**: Simple `RandomAffine` and `RandomHorizontalFlip`.\n",
    "*   **Loss Function**: `DiceBCELoss`.\n",
    "*   **Optimizer**: Adam.\n",
    "*   **Learning Rate**: 1e-4 (constant).\n",
    "*   **Epochs**: **50**.\n",
    "*   **Batch Size**: 1.\n",
    "*   **Image Size**: 256x256.\n",
    "*   **Data Split**: 80% train / 20% validation, split by patient ID."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5be50a81",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- Imports and Setup ---\n",
    "import os\n",
    "import pandas as pd\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "from torch.utils.data import DataLoader, Subset\n",
    "from torch.optim import Adam\n",
    "from tqdm import tqdm\n",
    "import matplotlib.pyplot as plt\n",
    "import sys\n",
    "\n",
    "project_root = os.path.abspath('..')\n",
    "if project_root not in sys.path:\n",
    "    sys.path.append(project_root)\n",
    "\n",
    "from src.data_loader import UterusDatasetWithPreprocessing \n",
    "from src.models import AttentionUNet\n",
    "\n",
    "# --- Configuration ---\n",
    "manifest_path = '../data/d2_manifest_t2fs_ovary_eligible.csv'\n",
    "image_size = 256\n",
    "batch_size = 1\n",
    "num_epochs = 50 # <--- CHANGE: Train for 50 epochs\n",
    "lr = 1e-4\n",
    "\n",
    "# --- Data Loading ---\n",
    "print(\"--- Loading Ovary Data with Custom Preprocessing ---\")\n",
    "train_full_dataset = UterusDatasetWithPreprocessing(manifest_path=manifest_path, image_size=image_size, augment=True)\n",
    "val_full_dataset = UterusDatasetWithPreprocessing(manifest_path=manifest_path, image_size=image_size, augment=False)\n",
    "patient_ids = train_full_dataset.manifest['patient_id'].unique()\n",
    "split_idx = int(len(patient_ids) * 0.8)\n",
    "train_ids, val_ids = patient_ids[:split_idx], patient_ids[split_idx:]\n",
    "train_indices = [i for i, sm in enumerate(train_full_dataset.slice_map) if train_full_dataset.manifest.loc[sm['patient_index'], 'patient_id'] in train_ids]\n",
    "val_indices = [i for i, sm in enumerate(val_full_dataset.slice_map) if val_full_dataset.manifest.loc[sm['patient_index'], 'patient_id'] in val_ids]\n",
    "train_dataset = Subset(train_full_dataset, train_indices)\n",
    "val_dataset = Subset(val_full_dataset, val_indices)\n",
    "train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)\n",
    "val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)\n",
    "print(f\"Data successfully split:\\nTraining patients: {len(train_ids)}, Validation patients: {len(val_ids)}\\nTraining slices: {len(train_dataset)}\\nValidation slices: {len(val_dataset)}\")\n",
    "\n",
    "# --- Metrics, Loss, and Training Functions ---\n",
    "def dice_score(preds, targets, epsilon=1e-6):\n",
    "    preds_flat, targets_flat = preds.view(-1), targets.view(-1)\n",
    "    intersection = (preds_flat * targets_flat).sum()\n",
    "    return (2. * intersection + epsilon) / (preds_flat.sum() + targets_flat.sum() + epsilon)\n",
    "\n",
    "class DiceBCELoss(nn.Module):\n",
    "    def __init__(self, weight=0.5):\n",
    "        super(DiceBCELoss, self).__init__()\n",
    "        self.weight = weight\n",
    "    def forward(self, inputs, targets, smooth=1):\n",
    "        inputs_sig = torch.sigmoid(inputs)\n",
    "        inputs_flat, targets_flat = inputs_sig.view(-1), targets.view(-1)\n",
    "        bce = F.binary_cross_entropy(inputs_flat, targets_flat, reduction='mean')\n",
    "        intersection = (inputs_flat * targets_flat).sum()\n",
    "        dice_loss = 1 - (2.*intersection + smooth)/(inputs_flat.sum() + targets_flat.sum() + smooth)\n",
    "        return bce * self.weight + dice_loss * (1 - self.weight)\n",
    "\n",
    "def train_one_epoch(model, loader, optimizer, criterion, device):\n",
    "    model.train()\n",
    "    running_loss = 0.0\n",
    "    for images, masks in tqdm(loader, desc=\"Training\"):\n",
    "        images, masks = images.to(device), masks.to(device)\n",
    "        optimizer.zero_grad()\n",
    "        outputs = model(images)\n",
    "        loss = criterion(outputs, masks)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        running_loss += loss.item() * images.size(0)\n",
    "    return running_loss / len(loader.dataset)\n",
    "\n",
    "def validate(model, loader, criterion, device):\n",
    "    model.eval()\n",
    "    running_loss, running_dice = 0.0, 0.0\n",
    "    with torch.no_grad():\n",
    "        for images, masks in tqdm(loader, desc=\"Validation\"):\n",
    "            images, masks = images.to(device), masks.to(device)\n",
    "            outputs = model(images)\n",
    "            loss = criterion(outputs, masks)\n",
    "            preds = torch.sigmoid(outputs) > 0.5\n",
    "            dice = dice_score(preds, masks)\n",
    "            running_loss += loss.item() * images.size(0)\n",
    "            running_dice += dice.item() * images.size(0)\n",
    "    return running_loss / len(loader.dataset), running_dice / len(loader.dataset)\n",
    "\n",
    "# --- Main Training Loop ---\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "print(f\"\\nUsing device: {device}\")\n",
    "\n",
    "model = AttentionUNet(n_channels=1, n_classes=1).to(device)\n",
    "optimizer = Adam(model.parameters(), lr=lr)\n",
    "criterion = DiceBCELoss()\n",
    "\n",
    "train_loss_history, val_loss_history, val_dice_history = [], [], []\n",
    "\n",
    "best_val_dice = -1.0\n",
    "best_epoch = -1\n",
    "model_save_path = \"../models/13_attn_unet_prep_long_best.pth\" # <--- CHANGE: New save path\n",
    "os.makedirs(os.path.dirname(model_save_path), exist_ok=True)\n",
    "\n",
    "print(\"\\n--- Starting Attention U-Net w/ Preprocessing (50 Epochs) ---\")\n",
    "for epoch in range(num_epochs):\n",
    "    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)\n",
    "    val_loss, val_dice = validate(model, val_loader, criterion, device)\n",
    "    train_loss_history.append(train_loss); val_loss_history.append(val_loss); val_dice_history.append(val_dice)\n",
    "    print(f\"Epoch {epoch+1}/{num_epochs} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}\")\n",
    "\n",
    "    if val_dice > best_val_dice:\n",
    "        best_val_dice = val_dice\n",
    "        best_epoch = epoch + 1\n",
    "        torch.save(model.state_dict(), model_save_path)\n",
    "        print(f\"  -> New best model saved at epoch {best_epoch} with Val Dice: {best_val_dice:.4f}\")\n",
    "\n",
    "print(\"--- Finished Training ---\")\n",
    "print(f\"Best model was from epoch {best_epoch} with a validation Dice score of {best_val_dice:.4f}\")\n",
    "print(f\"Model saved to {model_save_path}\\n\")\n",
    "\n",
    "# --- Visualization ---\n",
    "plt.figure(figsize=(15, 6))\n",
    "plt.subplot(1, 2, 1)\n",
    "plt.plot(range(1, num_epochs + 1), train_loss_history, label='Training Loss', marker='.')\n",
    "plt.plot(range(1, num_epochs + 1), val_loss_history, label='Validation Loss', marker='.')\n",
    "plt.title('Training and Validation Loss (50 Epochs)')\n",
    "plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)\n",
    "\n",
    "plt.subplot(1, 2, 2)\n",
    "plt.plot(range(1, num_epochs + 1), val_dice_history, label='Validation Dice Score', color='green', marker='.')\n",
    "plt.title('Validation Dice Score (50 Epochs)')\n",
    "plt.xlabel('Epochs'); plt.ylabel('Dice Score')\n",
    "plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Dice @ Epoch {best_epoch}')\n",
    "plt.legend(); plt.grid(True)\n",
    "plt.suptitle('Attention U-Net w/ Custom Preprocessing Results (50 Epochs)', fontsize=16)\n",
    "plt.tight_layout(rect=[0, 0.03, 1, 0.95])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dd2936d7",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- Sanity Check: Visualize Predictions from Best Model ---\n",
    "import numpy as np\n",
    "\n",
    "# --- Configuration for Analysis ---\n",
    "num_samples_to_show = 5\n",
    "\n",
    "# --- Load the BEST Model We Just Saved ---\n",
    "# The model_save_path variable is inherited from the cell above\n",
    "analysis_model = AttentionUNet(n_channels=1, n_classes=1).to(device)\n",
    "try:\n",
    "    analysis_model.load_state_dict(torch.load(model_save_path))\n",
    "    analysis_model.eval()\n",
    "    print(f\"Successfully loaded best model from epoch {best_epoch} for analysis.\")\n",
    "except FileNotFoundError:\n",
    "    print(f\"FATAL ERROR: Model file not found at {model_save_path}. Make sure the training cell ran correctly.\")\n",
    "    exit()\n",
    "\n",
    "# --- The val_loader is also inherited from the cell above ---\n",
    "print(f\"Visualizing {num_samples_to_show} sample predictions on the validation set...\")\n",
    "\n",
    "# Note: The \"Input Image\" shown here will be AFTER preprocessing\n",
    "fig, axes = plt.subplots(num_samples_to_show, 3, figsize=(12, num_samples_to_show * 4))\n",
    "fig.suptitle(f\"Predictions from Best Model (Epoch {best_epoch})\", fontsize=16)\n",
    "\n",
    "with torch.no_grad():\n",
    "    for i, (image, mask) in enumerate(val_loader):\n",
    "        if i >= num_samples_to_show:\n",
    "            break\n",
    "        image, mask = image.to(device), mask.to(device)\n",
    "        output = analysis_model(image)\n",
    "        pred_mask = torch.sigmoid(output) > 0.5\n",
    "        \n",
    "        image_np = image.cpu().squeeze().numpy()\n",
    "        mask_np = mask.cpu().squeeze().numpy()\n",
    "        pred_mask_np = pred_mask.cpu().squeeze().numpy()\n",
    "        \n",
    "        axes[i, 0].imshow(image_np, cmap='gray')\n",
    "        axes[i, 0].set_title(f\"Sample {i+1}: Preprocessed Input\")\n",
    "        axes[i, 0].axis('off')\n",
    "        \n",
    "        axes[i, 1].imshow(mask_np, cmap='gray')\n",
    "        axes[i, 1].set_title(\"Ground Truth Mask\")\n",
    "        axes[i, 1].axis('off')\n",
    "        \n",
    "        axes[i, 2].imshow(pred_mask_np, cmap='gray')\n",
    "        axes[i, 2].set_title(\"Predicted Mask\")\n",
    "        axes[i, 2].axis('off')\n",
    "\n",
    "plt.tight_layout(rect=[0, 0.03, 1, 0.96])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "40b85b44",
   "metadata": {},
   "source": [
    "# Analysis of Longer Training Run (Experiment 04)\n",
    "\n",
    "### Summary\n",
    "This notebook extends Experiment 03, training the Attention U-Net with custom preprocessing for 50 epochs to observe its full learning trajectory.\n",
    "\n",
    "### Quantitative Findings\n",
    "- **Peak Performance**: The model achieved a best validation Dice score of **0.3161 at epoch 16**. This represents a marginal improvement over the 20-epoch run but is still significantly worse than our initial U-Net baseline (0.4264).\n",
    "- **Training Instability**: The validation loss curve shows large, erratic spikes, indicating that the model is unstable and fails to generalize consistently.\n",
    "- **Confirmed Overfitting**: After epoch 16, the validation Dice score generally degrades, confirming that longer training with the current setup does not lead to better results and eventually leads to overfitting.\n",
    "\n",
    "### Qualitative Findings (Visual Analysis)\n",
    "The predictions from the best model (epoch 16) show some improvement over models without preprocessing (fewer random blobs), but still exhibit major flaws:\n",
    "- **False Negatives**: The model frequently fails to detect the ovary at all (e.g., Sample 1).\n",
    "- **False Positives**: The model still predicts extra, non-existent objects, showing continued confusion with surrounding tissues (e.g., Samples 2, 4, 5).\n",
    "\n",
    "### Conclusion & Next Steps\n",
    "This experiment confirms that a better architecture and preprocessing alone are insufficient. The model's poor performance and instability point to a fundamental problem with our loss function. The current `DiceBCELoss` struggles with the massive **class imbalance** between the tiny ovary and the vast background.\n",
    "\n",
    "The next logical step is to implement the specialized loss function used in the \"RAovSeg\" paper: the **Focal Tversky Loss**. This loss function is specifically designed for segmenting small objects by forcing the model to pay more attention to the difficult boundary pixels and penalizing false negatives more heavily."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "dlvr-project",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.23"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

## File: notebooks/14_data_deep_dive.ipynb/14_data_deep_dive.ipynb
```
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "ff0d6110",
   "metadata": {},
   "source": [
    "# Data Deep Dive: Investigating the Ground Truth Masks\n",
    "\n",
    "### Objective\n",
    "Before proceeding with more complex models, we need to build a stronger intuition for our ground truth data. This notebook serves as a visual audit to answer a key question:\n",
    "\n",
    "*   **Do any of the MRI slices in our eligible dataset contain two distinct ovaries?**\n",
    "\n",
    "Understanding this is crucial. If we only ever see one ovary per slice, the model is learning to identify a single object. If two are sometimes present, the problem is slightly different, and we need to be aware of it.\n",
    "\n",
    "### Methodology\n",
    "We will load our clean manifest of 56 eligible patients. We will then iterate through a large number of slices from the **entire dataset (both training and validation portions)** and plot a grid of the ground truth masks. Each mask will be labeled with its source `patient_id` and `slice_index` to ensure we are seeing a diverse sample."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3e0266d5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- Imports and Setup ---\n",
    "import os\n",
    "import pandas as pd\n",
    "import torch\n",
    "import matplotlib.pyplot as plt\n",
    "import sys\n",
    "import numpy as np\n",
    "\n",
    "# Add project root to path\n",
    "project_root = os.path.abspath('..')\n",
    "if project_root not in sys.path:\n",
    "    sys.path.append(project_root)\n",
    "\n",
    "from src.data_loader import UterusDataset\n",
    "\n",
    "# --- Configuration ---\n",
    "manifest_path = '../data/d2_manifest_t2fs_ovary_eligible.csv'\n",
    "image_size = 256\n",
    "num_patients_to_show = 36 # How many unique patients to visualize\n",
    "\n",
    "# --- Load the FULL eligible dataset ---\n",
    "full_dataset = UterusDataset(manifest_path=manifest_path, image_size=image_size, augment=False)\n",
    "\n",
    "print(f\"Total eligible patients with ovaries: {len(full_dataset.manifest)}\")\n",
    "print(f\"Displaying one sample slice from the first {num_patients_to_show} unique patients...\")\n",
    "\n",
    "# --- Visualize a Grid of Masks from DIFFERENT Patients ---\n",
    "fig, axes = plt.subplots(6, 6, figsize=(15, 15))\n",
    "fig.suptitle(\"Ground Truth Ovary Masks from Different Patients\", fontsize=20)\n",
    "axes = axes.flatten()\n",
    "\n",
    "# Get a list of unique patient IDs from the manifest\n",
    "unique_patient_ids = full_dataset.manifest['patient_id'].unique()\n",
    "\n",
    "# Keep track of which patients we've already plotted\n",
    "plotted_patients = set()\n",
    "slice_counter = 0\n",
    "plot_idx = 0\n",
    "\n",
    "# Iterate through the entire slice map to find one slice per patient\n",
    "while plot_idx < num_patients_to_show and slice_counter < len(full_dataset.slice_map):\n",
    "    slice_info = full_dataset.slice_map[slice_counter]\n",
    "    patient_id = full_dataset.manifest.iloc[slice_info['patient_index']]['patient_id']\n",
    "    \n",
    "    # If we haven't plotted this patient yet, plot their mask\n",
    "    if patient_id not in plotted_patients:\n",
    "        _, mask = full_dataset[slice_counter]\n",
    "        mask_np = mask.cpu().squeeze().numpy()\n",
    "        \n",
    "        ax = axes[plot_idx]\n",
    "        ax.imshow(mask_np, cmap='gray')\n",
    "        ax.set_title(f\"Patient: {patient_id}\")\n",
    "        ax.axis('off')\n",
    "        \n",
    "        plotted_patients.add(patient_id)\n",
    "        plot_idx += 1\n",
    "        \n",
    "    slice_counter += 1\n",
    "\n",
    "\n",
    "# Hide any unused subplots\n",
    "for j in range(plot_idx, len(axes)):\n",
    "    axes[j].axis('off')\n",
    "\n",
    "plt.tight_layout(rect=[0, 0.03, 1, 0.96])\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c04e5390",
   "metadata": {},
   "source": [
    "# Data Deep Dive: Analysis of Ground Truth Masks\n",
    "\n",
    "### Objective\n",
    "This notebook was created to investigate a key characteristic of our dataset. We wanted to confirm whether the ground truth masks typically contain one or two ovaries per slice, as this fundamentally impacts our modeling strategy.\n",
    "\n",
    "### Methodology\n",
    "We visualized 36 ground truth ovary masks, sampling one representative slice from 36 unique patients across the entire eligible dataset (both training and validation sets). This ensures a broad and unbiased view of the data's structure.\n",
    "\n",
    "### Key Finding\n",
    "The visual audit confirms that the ground truth data for this task almost exclusively consists of **single-object segmentation masks**. In our extensive sample, no slices containing two distinct, separate ovaries were observed.\n",
    "\n",
    "### Interpretation and Implications\n",
    "1.  **Confirmation of Task Difficulty**: This finding highlights the difficulty of the task. The target object is often singular, small, and variable in shape, making it a classic \"needle in a haystack\" problem for the model.\n",
    "2.  **Justification for the Paper's Pipeline**: This observation strongly supports the two-stage pipeline used in the \"RAovSeg\" paper. The first stage (`ResClass`) is critical for identifying the few slices that contain *any* ovary, while the second stage (`AttUSeg`, which we are building) can focus on the simpler task of segmenting the single object it is given.\n",
    "3.  **Confidence to Proceed**: We can now be confident that our current experimental setup, which trains a model to segment a single object per image, is the correct approach. We do not need to implement logic for separating multiple instances at this stage.\n",
    "\n",
    "This deep dive validates our direction and provides a solid foundation for the next experiments."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "dlvr-project",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.23"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

## File: notebooks/15_attention_unet_focal_tversky.ipynb/15_attention_unet_focal_tversky.ipynb
```
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "2c653f90",
   "metadata": {},
   "source": [
    "# Experiment 05: Attention U-Net with Focal Tversky Loss\n",
    "\n",
    "This notebook introduces the specialized loss function from the RAovSeg paper. We replace our generic `DiceBCELoss` with `FocalTverskyLoss`, which is designed to handle the class imbalance inherent in segmenting small objects like ovaries.\n",
    "\n",
    "### **Model Configuration**\n",
    "\n",
    "*   **Objective**: Evaluate the impact of the Focal Tversky loss function.\n",
    "*   **Model Architecture**: Attention U-Net.\n",
    "*   **Dataset**: D2_TCPW, eligible patients.\n",
    "*   **Preprocessing**: RAovSeg custom preprocessing.\n",
    "*   **Data Augmentation**: Simple `RandomAffine` and `RandomHorizontalFlip`.\n",
    "*   **Loss Function**: **Focal Tversky Loss (alpha=0.7, beta=0.3, gamma=4/3)**.\n",
    "*   **Optimizer**: Adam.\n",
    "*   **Learning Rate**: 1e-4 (constant).\n",
    "*   **Epochs**: **50**.\n",
    "*   **Batch Size**: 1.\n",
    "*   **Image Size**: 256x256.\n",
    "*   **Data Split**: 80% train / 20% validation, split by patient ID."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b5851364",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- Imports and Setup ---\n",
    "import os\n",
    "import pandas as pd\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "from torch.utils.data import DataLoader, Subset\n",
    "from torch.optim import Adam\n",
    "from tqdm import tqdm\n",
    "import matplotlib.pyplot as plt\n",
    "import sys\n",
    "import numpy as np\n",
    "\n",
    "project_root = os.path.abspath('..')\n",
    "if project_root not in sys.path:\n",
    "    sys.path.append(project_root)\n",
    "\n",
    "from src.data_loader import UterusDatasetWithPreprocessing \n",
    "from src.models import AttentionUNet\n",
    "from src.losses import FocalTverskyLoss\n",
    "\n",
    "# --- Configuration ---\n",
    "manifest_path = '../data/d2_manifest_t2fs_ovary_eligible.csv'\n",
    "image_size = 256\n",
    "batch_size = 1\n",
    "num_epochs = 50\n",
    "lr = 1e-4\n",
    "\n",
    "# --- Data Loading ---\n",
    "print(\"--- Loading Ovary Data with Custom Preprocessing ---\")\n",
    "train_full_dataset = UterusDatasetWithPreprocessing(manifest_path=manifest_path, image_size=image_size, augment=True)\n",
    "val_full_dataset = UterusDatasetWithPreprocessing(manifest_path=manifest_path, image_size=image_size, augment=False)\n",
    "patient_ids = train_full_dataset.manifest['patient_id'].unique()\n",
    "split_idx = int(len(patient_ids) * 0.8)\n",
    "train_ids, val_ids = patient_ids[:split_idx], patient_ids[split_idx:]\n",
    "train_indices = [i for i, sm in enumerate(train_full_dataset.slice_map) if train_full_dataset.manifest.loc[sm['patient_index'], 'patient_id'] in train_ids]\n",
    "val_indices = [i for i, sm in enumerate(val_full_dataset.slice_map) if val_full_dataset.manifest.loc[sm['patient_index'], 'patient_id'] in val_ids]\n",
    "train_dataset = Subset(train_full_dataset, train_indices)\n",
    "val_dataset = Subset(val_full_dataset, val_indices)\n",
    "train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)\n",
    "val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)\n",
    "print(f\"Data successfully split:\\nTraining patients: {len(train_ids)}, Validation patients: {len(val_ids)}\\nTraining slices: {len(train_dataset)}\\nValidation slices: {len(val_dataset)}\")\n",
    "\n",
    "# --- Metrics, Loss, and Training Functions ---\n",
    "def dice_score(preds, targets, epsilon=1e-6):\n",
    "    preds_flat, targets_flat = preds.view(-1), targets.view(-1)\n",
    "    intersection = (preds_flat * targets_flat).sum()\n",
    "    return (2. * intersection + epsilon) / (preds_flat.sum() + targets_flat.sum() + epsilon)\n",
    "\n",
    "def train_one_epoch(model, loader, optimizer, criterion, device):\n",
    "    model.train()\n",
    "    running_loss = 0.0\n",
    "    for images, masks in tqdm(loader, desc=\"Training\"):\n",
    "        images, masks = images.to(device), masks.to(device)\n",
    "        optimizer.zero_grad()\n",
    "        outputs = model(images)\n",
    "        loss = criterion(outputs, masks)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        running_loss += loss.item() * images.size(0)\n",
    "    return running_loss / len(loader.dataset)\n",
    "\n",
    "def validate(model, loader, criterion, device):\n",
    "    model.eval()\n",
    "    running_loss, running_dice = 0.0, 0.0\n",
    "    with torch.no_grad():\n",
    "        for images, masks in tqdm(loader, desc=\"Validation\"):\n",
    "            images, masks = images.to(device), masks.to(device)\n",
    "            outputs = model(images)\n",
    "            loss = criterion(outputs, masks)\n",
    "            preds = torch.sigmoid(outputs) > 0.5\n",
    "            dice = dice_score(preds, masks)\n",
    "            running_loss += loss.item() * images.size(0)\n",
    "            running_dice += dice.item() * images.size(0)\n",
    "    return running_loss / len(loader.dataset), running_dice / len(loader.dataset)\n",
    "\n",
    "# --- Main Training Loop ---\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "print(f\"\\nUsing device: {device}\")\n",
    "\n",
    "model = AttentionUNet(n_channels=1, n_classes=1).to(device)\n",
    "optimizer = Adam(model.parameters(), lr=lr)\n",
    "criterion = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=4/3)\n",
    "\n",
    "train_loss_history, val_loss_history, val_dice_history = [], [], []\n",
    "\n",
    "best_val_dice = -1.0\n",
    "best_epoch = -1\n",
    "model_save_path = \"../models/15_attn_unet_focal_tversky_best.pth\"\n",
    "os.makedirs(os.path.dirname(model_save_path), exist_ok=True)\n",
    "\n",
    "print(\"\\n--- Starting Training with Focal Tversky Loss (50 Epochs) ---\")\n",
    "for epoch in range(num_epochs):\n",
    "    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)\n",
    "    val_loss, val_dice = validate(model, val_loader, criterion, device)\n",
    "    train_loss_history.append(train_loss); val_loss_history.append(val_loss); val_dice_history.append(val_dice)\n",
    "    print(f\"Epoch {epoch+1}/{num_epochs} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}\")\n",
    "\n",
    "    if val_dice > best_val_dice:\n",
    "        best_val_dice = val_dice\n",
    "        best_epoch = epoch + 1\n",
    "        torch.save(model.state_dict(), model_save_path)\n",
    "        print(f\"  -> New best model saved at epoch {best_epoch} with Val Dice: {best_val_dice:.4f}\")\n",
    "\n",
    "print(\"--- Finished Training ---\")\n",
    "print(f\"Best model was from epoch {best_epoch} with a validation Dice score of {best_val_dice:.4f}\")\n",
    "print(f\"Model saved to {model_save_path}\\n\")\n",
    "\n",
    "# --- Visualization ---\n",
    "plt.figure(figsize=(15, 6))\n",
    "plt.subplot(1, 2, 1)\n",
    "plt.plot(range(1, num_epochs + 1), train_loss_history, label='Training Loss', marker='.')\n",
    "plt.plot(range(1, num_epochs + 1), val_loss_history, label='Validation Loss', marker='.')\n",
    "plt.title('Training and Validation Loss (Focal Tversky)')\n",
    "plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)\n",
    "\n",
    "plt.subplot(1, 2, 2)\n",
    "plt.plot(range(1, num_epochs + 1), val_dice_history, label='Validation Dice Score', color='green', marker='.')\n",
    "plt.title('Validation Dice Score (Focal Tversky)')\n",
    "plt.xlabel('Epochs'); plt.ylabel('Dice Score')\n",
    "plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Dice @ Epoch {best_epoch}')\n",
    "plt.legend(); plt.grid(True)\n",
    "plt.suptitle('Attention U-Net w/ Focal Tversky Loss Results (50 Epochs)', fontsize=16)\n",
    "plt.tight_layout(rect=[0, 0.03, 1, 0.95])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b708619e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- Sanity Check: Visualize Predictions from Best Model ---\n",
    "\n",
    "# --- Configuration for Analysis ---\n",
    "num_samples_to_show = 5\n",
    "\n",
    "# --- Load the BEST Model We Just Saved ---\n",
    "# Variables like 'model_save_path', 'best_epoch', and 'val_loader' are inherited from the cell above\n",
    "analysis_model = AttentionUNet(n_channels=1, n_classes=1).to(device)\n",
    "try:\n",
    "    analysis_model.load_state_dict(torch.load(model_save_path))\n",
    "    analysis_model.eval()\n",
    "    print(f\"\\n--- Analysis: Loading best model from epoch {best_epoch} ---\")\n",
    "except FileNotFoundError:\n",
    "    print(f\"FATAL ERROR: Model file not found at {model_save_path}. Make sure the training cell ran correctly.\")\n",
    "    exit()\n",
    "\n",
    "print(f\"Visualizing {num_samples_to_show} sample predictions on the validation set...\")\n",
    "\n",
    "fig, axes = plt.subplots(num_samples_to_show, 3, figsize=(12, num_samples_to_show * 4))\n",
    "fig.suptitle(f\"Predictions from Best Model (Epoch {best_epoch})\", fontsize=16)\n",
    "\n",
    "with torch.no_grad():\n",
    "    for i, (image, mask) in enumerate(val_loader):\n",
    "        if i >= num_samples_to_show:\n",
    "            break\n",
    "        image, mask = image.to(device), mask.to(device)\n",
    "        output = analysis_model(image)\n",
    "        pred_mask = torch.sigmoid(output) > 0.5\n",
    "        \n",
    "        image_np = image.cpu().squeeze().numpy()\n",
    "        mask_np = mask.cpu().squeeze().numpy()\n",
    "        pred_mask_np = pred_mask.cpu().squeeze().numpy()\n",
    "        \n",
    "        axes[i, 0].imshow(image_np, cmap='gray')\n",
    "        axes[i, 0].set_title(f\"Sample {i+1}: Preprocessed Input\")\n",
    "        axes[i, 0].axis('off')\n",
    "        \n",
    "        axes[i, 1].imshow(mask_np, cmap='gray')\n",
    "        axes[i, 1].set_title(\"Ground Truth Mask\")\n",
    "        axes[i, 1].axis('off')\n",
    "        \n",
    "        axes[i, 2].imshow(pred_mask_np, cmap='gray')\n",
    "        axes[i, 2].set_title(\"Predicted Mask\")\n",
    "        axes[i, 2].axis('off')\n",
    "\n",
    "plt.tight_layout(rect=[0, 0.03, 1, 0.96])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fd79a402",
   "metadata": {},
   "source": [
    "# Analysis of Focal Tversky Loss Experiment (Experiment 05)\n",
    "\n",
    "### Summary\n",
    "This notebook documents our experiment using the Attention U-Net with custom preprocessing and the specialized Focal Tversky loss function, as described in the RAovSeg paper. The model was trained for 50 epochs to observe its learning behavior with this new loss function.\n",
    "\n",
    "### Quantitative Findings\n",
    "- **Peak Performance**: The model achieved a best validation Dice score of **0.3161 at epoch 16**. This result did not surpass our initial U-Net baseline score of 0.4264.\n",
    "- **Training Dynamics**: Both the validation loss and Dice score curves are highly volatile. The validation loss plateaus after approximately 15 epochs, indicating that the model stops generalizing effectively around that point, even as the training loss continues to decrease. This confirms that overfitting is still a significant issue.\n",
    "\n",
    "### Qualitative Findings (Visual Analysis)\n",
    "The visual predictions from the best model (epoch 16) show a noticeable improvement in one key area compared to previous experiments:\n",
    "- **Reduced False Negatives**: The model is more consistent at identifying a segmentation mask in the correct location. It fails to predict an ovary entirely (like in Sample 1) less often than the `DiceBCELoss` model.\n",
    "- **Boundary Precision Problem**: While the model is better at finding the target, it struggles with defining its precise boundaries. The predicted masks are often larger and more \"blob-like\" than the ground truth, and frequently include small, disconnected false positive regions nearby.\n",
    "\n",
    "### Conclusion & Next Steps\n",
    "The Focal Tversky loss helped the model to be more aggressive in finding potential ovary regions, successfully reducing the number of missed detections (false negatives). However, it did not solve the precision problem and resulted in very noisy predictions.\n",
    "\n",
    "The visual results strongly suggest that the next logical step is to implement the **post-processing** method described in the paper. The small, scattered blobs in our predictions are exactly the kind of noise that morphological operations (like binary closing) and connected component analysis are designed to remove. This should clean up our predictions and significantly improve the final Dice score."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "dlvr-project",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.23"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

## File: notebooks/16_analyze_with_postprocessing.ipynb/16_analyze_with_postprocessing.ipynb
```
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "231b7037",
   "metadata": {},
   "source": [
    "# Experiment 06: Analysis with Post-processing\n",
    "\n",
    "### Summary\n",
    "This notebook evaluates the impact of the post-processing step described in the \"RAovSeg\" paper. We take the best model from our previous long-training experiment (`13_...ipynb`) and apply post-processing (binary closing + keeping the largest connected component) to its raw predictions on the validation set.\n",
    "\n",
    "### **Methodology**\n",
    "\n",
    "*   **Objective**: Quantify the performance improvement gained by applying post-processing to the output of our best model so far.\n",
    "*   **Model Used**: The saved `AttentionUNet` from Experiment 04 (trained for 50 epochs with custom preprocessing and Focal Tversky Loss).\n",
    "*   **Analysis Steps**:\n",
    "    1.  Load the pre-trained model.\n",
    "    2.  Iterate over the entire validation set.\n",
    "    3.  For each sample, calculate the Dice score on the raw prediction.\n",
    "    4.  Apply the `postprocess_` function to the raw prediction.\n",
    "    5.  Calculate the Dice score on the post-processed prediction.\n",
    "    6.  Report the average Dice scores for both \"before\" and \"after\" to measure the impact.\n",
    "    7.  Visualize a few examples to confirm the effect."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1101fd33",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- Imports and Setup ---\n",
    "import os\n",
    "import pandas as pd\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "from torch.utils.data import DataLoader, Subset\n",
    "from tqdm import tqdm\n",
    "import matplotlib.pyplot as plt\n",
    "import sys\n",
    "import numpy as np\n",
    "\n",
    "project_root = os.path.abspath('..')\n",
    "if project_root not in sys.path:\n",
    "    sys.path.append(project_root)\n",
    "\n",
    "from src.data_loader import UterusDatasetWithPreprocessing \n",
    "from src.models import AttentionUNet\n",
    "from src.RAovSeg_tools import postprocess_ # <--- Import the post-processing function\n",
    "\n",
    "# --- Configuration ---\n",
    "model_path = \"../models/15_attn_unet_focal_tversky_best.pth\" # <--- Path to the model from the last experiment\n",
    "manifest_path = '../data/d2_manifest_t2fs_ovary_eligible.csv'\n",
    "image_size = 256\n",
    "num_samples_to_show = 5\n",
    "\n",
    "# --- Load Model ---\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "model = AttentionUNet(n_channels=1, n_classes=1).to(device)\n",
    "\n",
    "try:\n",
    "    model.load_state_dict(torch.load(model_path))\n",
    "    model.eval()\n",
    "    print(f\"Successfully loaded model from {model_path}\")\n",
    "except FileNotFoundError:\n",
    "    print(f\"FATAL ERROR: Model file not found at {model_path}. Please run the training in notebook 15.\")\n",
    "    exit()\n",
    "\n",
    "# --- Load Validation Data ---\n",
    "val_full_dataset = UterusDatasetWithPreprocessing(manifest_path=manifest_path, image_size=image_size, augment=False)\n",
    "patient_ids = val_full_dataset.manifest['patient_id'].unique()\n",
    "split_idx = int(len(patient_ids) * 0.8)\n",
    "val_ids = patient_ids[split_idx:]\n",
    "val_indices = [i for i, sm in enumerate(val_full_dataset.slice_map) if val_full_dataset.manifest.loc[sm['patient_index'], 'patient_id'] in val_ids]\n",
    "val_dataset = Subset(val_full_dataset, val_indices)\n",
    "val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)\n",
    "print(f\"Loaded {len(val_dataset)} validation slices for analysis.\\n\")\n",
    "\n",
    "# --- Quantitative Analysis: Calculate Dice Scores Before and After Post-processing ---\n",
    "\n",
    "def dice_score(preds, targets, epsilon=1e-6):\n",
    "    preds_flat, targets_flat = preds.view(-1), targets.view(-1)\n",
    "    intersection = (preds_flat * targets_flat).sum()\n",
    "    return (2. * intersection + epsilon) / (preds_flat.sum() + targets_flat.sum() + epsilon)\n",
    "\n",
    "dice_scores_before = []\n",
    "dice_scores_after = []\n",
    "\n",
    "with torch.no_grad():\n",
    "    for image, mask in tqdm(val_loader, desc=\"Analyzing Validation Set\"):\n",
    "        image, mask = image.to(device), mask.to(device)\n",
    "        \n",
    "        # 1. Get raw prediction\n",
    "        output = model(image)\n",
    "        pred_mask_raw = torch.sigmoid(output) > 0.5\n",
    "        \n",
    "        # 2. Calculate Dice score BEFORE post-processing\n",
    "        dice_before = dice_score(pred_mask_raw, mask)\n",
    "        dice_scores_before.append(dice_before.cpu().item())\n",
    "        \n",
    "        # 3. Apply post-processing\n",
    "        pred_mask_raw_np = pred_mask_raw.cpu().squeeze().numpy()\n",
    "        pred_mask_post_np = postprocess_(pred_mask_raw_np)\n",
    "        \n",
    "        # 4. Calculate Dice score AFTER post-processing\n",
    "        pred_mask_post_tensor = torch.from_numpy(pred_mask_post_np).unsqueeze(0).unsqueeze(0).to(device)\n",
    "        dice_after = dice_score(pred_mask_post_tensor, mask)\n",
    "        dice_scores_after.append(dice_after.cpu().item())\n",
    "\n",
    "# Calculate and print average scores\n",
    "avg_dice_before = np.mean(dice_scores_before)\n",
    "avg_dice_after = np.mean(dice_scores_after)\n",
    "\n",
    "print(\"\\n--- Quantitative Analysis Results ---\")\n",
    "print(f\"Average Validation Dice Score BEFORE Post-processing: {avg_dice_before:.4f}\")\n",
    "print(f\"Average Validation Dice Score AFTER Post-processing:  {avg_dice_after:.4f}\")\n",
    "print(f\"Improvement: {avg_dice_after - avg_dice_before:+.4f}\")\n",
    "\n",
    "\n",
    "# --- Qualitative Analysis: Visualize Predictions ---\n",
    "print(f\"\\nVisualizing {num_samples_to_show} sample predictions...\")\n",
    "\n",
    "fig, axes = plt.subplots(num_samples_to_show, 4, figsize=(16, num_samples_to_show * 4))\n",
    "fig.suptitle(\"Effect of Post-processing on Predictions\", fontsize=16)\n",
    "\n",
    "with torch.no_grad():\n",
    "    for i, (image, mask) in enumerate(val_loader):\n",
    "        if i >= num_samples_to_show:\n",
    "            break\n",
    "        \n",
    "        image, mask = image.to(device), mask.to(device)\n",
    "        output = model(image)\n",
    "        pred_mask_raw = torch.sigmoid(output) > 0.5\n",
    "        \n",
    "        # Post-process for visualization\n",
    "        pred_mask_post = postprocess_(pred_mask_raw.cpu().squeeze().numpy())\n",
    "\n",
    "        # Convert all to numpy for plotting\n",
    "        image_np = image.cpu().squeeze().numpy()\n",
    "        mask_np = mask.cpu().squeeze().numpy()\n",
    "        pred_mask_raw_np = pred_mask_raw.cpu().squeeze().numpy()\n",
    "        \n",
    "        # Plotting\n",
    "        axes[i, 0].imshow(image_np, cmap='gray')\n",
    "        axes[i, 0].set_title(f\"Sample {i+1}: Input\")\n",
    "        axes[i, 0].axis('off')\n",
    "        \n",
    "        axes[i, 1].imshow(mask_np, cmap='gray')\n",
    "        axes[i, 1].set_title(\"Ground Truth\")\n",
    "        axes[i, 1].axis('off')\n",
    "\n",
    "        axes[i, 2].imshow(pred_mask_raw_np, cmap='gray')\n",
    "        axes[i, 2].set_title(\"Prediction (Before)\")\n",
    "        axes[i, 2].axis('off')\n",
    "\n",
    "        axes[i, 3].imshow(pred_mask_post, cmap='gray')\n",
    "        axes[i, 3].set_title(\"Prediction (After)\")\n",
    "        axes[i, 3].axis('off')\n",
    "\n",
    "plt.tight_layout(rect=[0, 0.03, 1, 0.96])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8bc3d687",
   "metadata": {},
   "source": [
    "# Analysis of Post-processing (Experiment 06)\n",
    "\n",
    "### Summary\n",
    "This notebook analyzes the effect of applying the paper's post-processing method (binary closing and keeping the largest connected component) to the predictions of our best model from Experiment 05.\n",
    "\n",
    "### Quantitative Findings\n",
    "- **Performance Change**: The post-processing step resulted in a slight **decrease** in the average validation Dice score.\n",
    "    - Average Dice Score (Before): **0.3090**\n",
    "    - Average Dice Score (After):  **0.2988**\n",
    "    - Improvement: **-0.0102**\n",
    "\n",
    "### Qualitative Findings (Visual Analysis)\n",
    "The visual analysis reveals the reason for the drop in performance and highlights a key weakness in our current model.\n",
    "- **Identified Problem**: The model often produces multiple predictions for a single slice, and it is frequently more confident (i.e., predicts a larger area) for an incorrect region than for the true ovary.\n",
    "- **Failure Mode**: The post-processing logic, which keeps only the single largest predicted region, therefore often **discards the correct, smaller true positive** while keeping a larger false positive (e.g., Samples 2, 3).\n",
    "- **Success Case**: In cases where the model's largest prediction was correct and it also produced smaller, spurious noise, the post-processing worked as intended and successfully cleaned up the prediction (e.g., Sample 4).\n",
    "\n",
    "### Conclusion & Next Steps\n",
    "This experiment is highly valuable. It demonstrates that while the post-processing logic can clean up minor noise, it is not a solution for a model that is fundamentally uncertain. The core issue is that the model is still being forced to segment slices where it gets confused by other tissues that mimic the appearance of an ovary.\n",
    "\n",
    "This provides the definitive motivation for our next step: implementing the **`ResClass` slice selection classifier**. By first training a model whose only job is to identify which slices are worth segmenting, we can prevent the segmentation model from ever seeing these confusing \"negative\" slices. This should dramatically reduce the occurrence of these large, confident false positives and allow the post-processing step to be much more effective."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "dlvr-project",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.23"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

## File: notebooks/17_resnet_classifier.ipynb/17_resnet_classifier.ipynb
```
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "7e00acff",
   "metadata": {},
   "source": [
    "# Experiment 07: ResNet Slice Classifier (ResClass)\n",
    "\n",
    "This notebook implements the first stage of the RAovSeg pipeline: the slice selection classifier. The goal is to train a ResNet-18 model to distinguish between MRI slices that contain an ovary and those that do not.\n",
    "\n",
    "### **Model Configuration**\n",
    "\n",
    "*   **Objective**: Train a binary classifier to identify ovary-containing slices.\n",
    "*   **Model Architecture**: **ResNet-18** (from `torchvision.models`).\n",
    "*   **Dataset**: D2_TCPW, using **all slices** from eligible patients, with binary labels (1=ovary, 0=no ovary).\n",
    "*   **Preprocessing**: RAovSeg custom preprocessing.\n",
    "*   **Data Augmentation**: Simple `RandomAffine` and `RandomHorizontalFlip`.\n",
    "*   **Loss Function**: **`BCEWithLogitsLoss`** (standard for binary classification).\n",
    "*   **Optimizer**: Adam.\n",
    "*   **Learning Rate**: 1e-4 (constant).\n",
    "*   **Epochs**: 20.\n",
    "*   **Batch Size**: **16** (we can use a larger batch size for classification).\n",
    "*   **Image Size**: 256x256.\n",
    "*   **Data Split**: 80% train / 20% validation, split by patient ID.\n",
    "*   **Class Imbalance Strategy**: Subsampling negative examples to a 2:1 ratio (negative:positive)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8b3bffdc",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- Imports and Setup ---\n",
    "import os\n",
    "import pandas as pd\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "from torch.utils.data import DataLoader, Subset\n",
    "from torch.optim import Adam\n",
    "from tqdm import tqdm\n",
    "import matplotlib.pyplot as plt\n",
    "import sys\n",
    "import numpy as np\n",
    "import torchvision.models as models\n",
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score\n",
    "\n",
    "project_root = os.path.abspath('..')\n",
    "if project_root not in sys.path:\n",
    "    sys.path.append(project_root)\n",
    "\n",
    "# <--- Import our new Dataset class ---\n",
    "from src.data_loader import SliceClassifierDataset\n",
    "\n",
    "# --- Configuration ---\n",
    "manifest_path = '../data/d2_manifest_t2fs_ovary_eligible.csv'\n",
    "image_size = 256\n",
    "batch_size = 16 # We can use a larger batch size for classification\n",
    "num_epochs = 20\n",
    "lr = 1e-4\n",
    "\n",
    "# --- Data Loading ---\n",
    "print(\"--- Loading Full Slice Data for Classification ---\")\n",
    "# Use the new dataset class which loads positive and negative samples\n",
    "train_full_dataset = SliceClassifierDataset(manifest_path=manifest_path, image_size=image_size, augment=True)\n",
    "val_full_dataset = SliceClassifierDataset(manifest_path=manifest_path, image_size=image_size, augment=False)\n",
    "\n",
    "# This split is now on the slice level, but the underlying patient IDs are still separated by the dataset object creation.\n",
    "# For simplicity, we'll split the shuffled list of slices.\n",
    "num_slices = len(train_full_dataset.slice_data)\n",
    "split_idx = int(num_slices * 0.8)\n",
    "\n",
    "# Note: This is a simplified split. A more rigorous approach would split patient IDs first.\n",
    "# But since the data is shuffled, this is a reasonable starting point.\n",
    "train_dataset = Subset(train_full_dataset, range(split_idx))\n",
    "val_dataset = Subset(val_full_dataset, range(split_idx, num_slices))\n",
    "\n",
    "\n",
    "train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)\n",
    "val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)\n",
    "print(f\"Data successfully split:\\nTraining samples: {len(train_dataset)}\\nValidation samples: {len(val_dataset)}\")\n",
    "\n",
    "\n",
    "# --- Training and Validation Functions for CLASSIFICATION ---\n",
    "def train_one_epoch(model, loader, optimizer, criterion, device):\n",
    "    model.train()\n",
    "    running_loss = 0.0\n",
    "    all_preds, all_labels = [], []\n",
    "    for images, labels in tqdm(loader, desc=\"Training\"):\n",
    "        images, labels = images.to(device), labels.to(device).unsqueeze(1)\n",
    "        optimizer.zero_grad()\n",
    "        outputs = model(images)\n",
    "        loss = criterion(outputs, labels)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        \n",
    "        running_loss += loss.item() * images.size(0)\n",
    "        \n",
    "        preds = torch.sigmoid(outputs) > 0.5\n",
    "        all_preds.extend(preds.cpu().numpy())\n",
    "        all_labels.extend(labels.cpu().numpy())\n",
    "        \n",
    "    accuracy = accuracy_score(all_labels, all_preds)\n",
    "    return running_loss / len(loader.dataset), accuracy\n",
    "\n",
    "\n",
    "def validate(model, loader, criterion, device):\n",
    "    model.eval()\n",
    "    running_loss = 0.0\n",
    "    all_preds, all_labels = [], []\n",
    "    with torch.no_grad():\n",
    "        for images, labels in tqdm(loader, desc=\"Validation\"):\n",
    "            images, labels = images.to(device), labels.to(device).unsqueeze(1)\n",
    "            outputs = model(images)\n",
    "            loss = criterion(outputs, labels)\n",
    "            \n",
    "            running_loss += loss.item() * images.size(0)\n",
    "            \n",
    "            preds = torch.sigmoid(outputs) > 0.5\n",
    "            all_preds.extend(preds.cpu().numpy())\n",
    "            all_labels.extend(labels.cpu().numpy())\n",
    "            \n",
    "    accuracy = accuracy_score(all_labels, all_preds)\n",
    "    precision = precision_score(all_labels, all_preds, zero_division=0)\n",
    "    recall = recall_score(all_labels, all_preds, zero_division=0)\n",
    "    \n",
    "    return running_loss / len(loader.dataset), accuracy, precision, recall\n",
    "\n",
    "# --- Main Training Loop ---\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "print(f\"\\nUsing device: {device}\")\n",
    "\n",
    "# Load a pretrained ResNet-18 and adapt it for our single-channel, binary task\n",
    "model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)\n",
    "# Modify for single-channel input\n",
    "model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)\n",
    "# Modify for binary output\n",
    "model.fc = nn.Linear(model.fc.in_features, 1)\n",
    "model = model.to(device)\n",
    "\n",
    "optimizer = Adam(model.parameters(), lr=lr)\n",
    "criterion = nn.BCEWithLogitsLoss() # Standard loss for binary classification\n",
    "\n",
    "train_loss_history, val_loss_history, val_acc_history = [], [], []\n",
    "best_val_acc = -1.0\n",
    "best_epoch = -1\n",
    "model_save_path = \"../models/17_resclass_best.pth\"\n",
    "os.makedirs(os.path.dirname(model_save_path), exist_ok=True)\n",
    "\n",
    "print(\"\\n--- Starting ResNet Classifier (ResClass) Training ---\")\n",
    "for epoch in range(num_epochs):\n",
    "    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)\n",
    "    val_loss, val_acc, val_prec, val_recall = validate(model, val_loader, criterion, device)\n",
    "    \n",
    "    train_loss_history.append(train_loss)\n",
    "    val_loss_history.append(val_loss)\n",
    "    val_acc_history.append(val_acc)\n",
    "    \n",
    "    print(f\"Epoch {epoch+1}/{num_epochs} -> Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Precision: {val_prec:.4f}, Val Recall: {val_recall:.4f}\")\n",
    "\n",
    "    if val_acc > best_val_acc:\n",
    "        best_val_acc = val_acc\n",
    "        best_epoch = epoch + 1\n",
    "        torch.save(model.state_dict(), model_save_path)\n",
    "        print(f\"  -> New best model saved at epoch {best_epoch} with Val Acc: {best_val_acc:.4f}\")\n",
    "\n",
    "print(\"--- Finished Training ---\")\n",
    "print(f\"Best model was from epoch {best_epoch} with a validation accuracy of {best_val_acc:.4f}\")\n",
    "print(f\"Model saved to {model_save_path}\\n\")\n",
    "\n",
    "# --- Visualization ---\n",
    "plt.figure(figsize=(15, 6))\n",
    "plt.subplot(1, 2, 1)\n",
    "plt.plot(range(1, num_epochs + 1), train_loss_history, label='Training Loss', marker='.')\n",
    "plt.plot(range(1, num_epochs + 1), val_loss_history, label='Validation Loss', marker='.')\n",
    "plt.title('Training and Validation Loss (ResClass)')\n",
    "plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)\n",
    "\n",
    "plt.subplot(1, 2, 2)\n",
    "plt.plot(range(1, num_epochs + 1), val_acc_history, label='Validation Accuracy', color='green', marker='.')\n",
    "plt.title('Validation Accuracy (ResClass)')\n",
    "plt.xlabel('Epochs'); plt.ylabel('Accuracy')\n",
    "plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Acc @ Epoch {best_epoch}')\n",
    "plt.legend(); plt.grid(True)\n",
    "plt.suptitle('ResNet Classifier (ResClass) Results', fontsize=16)\n",
    "plt.tight_layout(rect=[0, 0.03, 1, 0.95])\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "dlvr-project",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.23"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

## File: notebooks/18_cyclical_lr.ipynb/18_cyclical_lr.ipynb
```
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "10557597",
   "metadata": {},
   "source": [
    "# Experiment 07: Attention U-Net with Cyclical Learning Rate (CLR)\n",
    "\n",
    "This notebook addresses the training instability observed in previous experiments. We will use a Cyclical Learning Rate (CLR) schedule instead of a constant learning rate. The goal is to help the optimizer escape poor local minima and find a more robust solution.\n",
    "\n",
    "### **Methodology**\n",
    "\n",
    "This is a two-part experiment:\n",
    "1.  **LR Range Test**: We first run a short training process where we linearly increase the learning rate from a very small to a large value. We plot the loss vs. the learning rate to identify the optimal range where the loss decreases most rapidly.\n",
    "2.  **Full Training**: We use the identified optimal range to train our full model for 50 epochs using a CLR scheduler.\n",
    "\n",
    "### **Model Configuration**\n",
    "\n",
    "*   **Objective**: Stabilize training and improve performance using a CLR schedule.\n",
    "*   **Model Architecture**: Attention U-Net.\n",
    "*   **Dataset**: D2_TCPW, eligible patients.\n",
    "*   **Preprocessing**: RAovSeg custom preprocessing.\n",
    "*   **Loss Function**: Focal Tversky Loss.\n",
    "*   **Optimizer**: Adam.\n",
    "*   **Learning Rate**: **Cyclical, varying between a min and max bound.**\n",
    "*   **Epochs**: 50.\n",
    "*   **Batch Size**: 16."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b8d18f77",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- Imports and Setup ---\n",
    "import os\n",
    "import pandas as pd\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "from torch.utils.data import DataLoader, Subset\n",
    "from torch.optim import Adam\n",
    "from torch.optim.lr_scheduler import LambdaLR\n",
    "from tqdm import tqdm\n",
    "import matplotlib.pyplot as plt\n",
    "import sys\n",
    "import numpy as np\n",
    "\n",
    "project_root = os.path.abspath('..')\n",
    "if project_root not in sys.path:\n",
    "    sys.path.append(project_root)\n",
    "\n",
    "from src.data_loader import UterusDatasetWithPreprocessing \n",
    "from src.models import AttentionUNet\n",
    "from src.losses import FocalTverskyLoss\n",
    "\n",
    "# --- Configuration for LR Range Test ---\n",
    "manifest_path = '../data/d2_manifest_t2fs_ovary_eligible.csv'\n",
    "image_size = 256\n",
    "batch_size = 16\n",
    "start_lr = 1e-7\n",
    "end_lr = 1e-1\n",
    "num_steps = 100 # Number of steps to increase the LR\n",
    "\n",
    "# --- Data Loading ---\n",
    "print(\"--- Loading Data for LR Range Test ---\")\n",
    "# Only need the training data for this test\n",
    "train_full_dataset = UterusDatasetWithPreprocessing(manifest_path=manifest_path, image_size=image_size, augment=True)\n",
    "patient_ids = train_full_dataset.manifest['patient_id'].unique()\n",
    "split_idx = int(len(patient_ids) * 0.8)\n",
    "train_ids = patient_ids[:split_idx]\n",
    "train_indices = [i for i, sm in enumerate(train_full_dataset.slice_map) if train_full_dataset.manifest.loc[sm['patient_index'], 'patient_id'] in train_ids]\n",
    "train_dataset = Subset(train_full_dataset, train_indices)\n",
    "train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)\n",
    "\n",
    "# --- LR Range Test Logic ---\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "print(f\"\\nUsing device: {device}\")\n",
    "\n",
    "model = AttentionUNet(n_channels=1, n_classes=1).to(device)\n",
    "optimizer = Adam(model.parameters(), lr=start_lr) # Start with a very small LR\n",
    "criterion = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=4/3)\n",
    "\n",
    "# Linearly increase LR from start_lr to end_lr over num_steps\n",
    "lr_lambda = lambda step: (end_lr / start_lr) ** (step / num_steps)\n",
    "scheduler = LambdaLR(optimizer, lr_lambda)\n",
    "\n",
    "learning_rates = []\n",
    "losses = []\n",
    "\n",
    "model.train()\n",
    "iterator = iter(train_loader)\n",
    "for step in tqdm(range(num_steps), desc=\"LR Range Test\"):\n",
    "    try:\n",
    "        images, masks = next(iterator)\n",
    "    except StopIteration:\n",
    "        iterator = iter(train_loader)\n",
    "        images, masks = next(iterator)\n",
    "\n",
    "    images, masks = images.to(device), masks.to(device)\n",
    "    \n",
    "    optimizer.zero_grad()\n",
    "    outputs = model(images)\n",
    "    loss = criterion(outputs, masks)\n",
    "    \n",
    "    # Break if loss explodes\n",
    "    if torch.isnan(loss) or loss > 4 * min(losses, default=1.0):\n",
    "        print(\"Loss exploded, stopping test.\")\n",
    "        break\n",
    "        \n",
    "    loss.backward()\n",
    "    optimizer.step()\n",
    "    \n",
    "    learning_rates.append(scheduler.get_last_lr()[0])\n",
    "    losses.append(loss.item())\n",
    "    \n",
    "    scheduler.step()\n",
    "\n",
    "# --- Plot the Results ---\n",
    "print(\"Plotting LR Range Test Results...\")\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.plot(learning_rates, losses)\n",
    "plt.xscale(\"log\")\n",
    "plt.xlabel(\"Learning Rate (log scale)\")\n",
    "plt.ylabel(\"Loss\")\n",
    "plt.title(\"Learning Rate Range Test\")\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "aa52a706",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- Imports for the full training run ---\n",
    "from torch.optim.lr_scheduler import CyclicLR\n",
    "# --- ADDED Imports that were missing ---\n",
    "import pandas as pd\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "from torch.utils.data import DataLoader, Subset\n",
    "from torch.optim import Adam\n",
    "from tqdm import tqdm\n",
    "import matplotlib.pyplot as plt\n",
    "import sys\n",
    "import numpy as np\n",
    "\n",
    "# --- Imports and Setup ---\n",
    "import os\n",
    "import pandas as pd\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "from torch.utils.data import DataLoader, Subset\n",
    "from torch.optim import Adam\n",
    "from torch.optim.lr_scheduler import LambdaLR\n",
    "from tqdm import tqdm\n",
    "import matplotlib.pyplot as plt\n",
    "import sys\n",
    "import numpy as np\n",
    "\n",
    "project_root = os.path.abspath('..')\n",
    "if project_root not in sys.path:\n",
    "    sys.path.append(project_root)\n",
    "\n",
    "from src.data_loader import UterusDatasetWithPreprocessing \n",
    "from src.models import AttentionUNet\n",
    "from src.losses import FocalTverskyLoss\n",
    "\n",
    "project_root = os.path.abspath('..')\n",
    "if project_root not in sys.path:\n",
    "    sys.path.append(project_root)\n",
    "\n",
    "from src.data_loader import UterusDatasetWithPreprocessing \n",
    "from src.models import AttentionUNet\n",
    "from src.losses import FocalTverskyLoss\n",
    "# ------------------------------------\n",
    "\n",
    "# --- Configuration for Full Training ---\n",
    "base_lr = 1e-3\n",
    "max_lr = 1e-2\n",
    "num_epochs = 50\n",
    "batch_size = 16\n",
    "image_size = 256\n",
    "manifest_path = '../data/d2_manifest_t2fs_ovary_eligible.csv' # <-- ADDED manifest path here\n",
    "\n",
    "# --- Data Loading (Now self-contained) ---\n",
    "print(\"--- Loading Data for Full Training Run ---\")\n",
    "# --- ADDED data loading back into this cell ---\n",
    "train_full_dataset = UterusDatasetWithPreprocessing(manifest_path=manifest_path, image_size=image_size, augment=True)\n",
    "val_full_dataset = UterusDatasetWithPreprocessing(manifest_path=manifest_path, image_size=image_size, augment=False)\n",
    "# -------------------------------------------\n",
    "patient_ids = val_full_dataset.manifest['patient_id'].unique()\n",
    "split_idx = int(len(patient_ids) * 0.8)\n",
    "train_ids, val_ids = patient_ids[:split_idx], patient_ids[split_idx:]\n",
    "train_indices = [i for i, sm in enumerate(val_full_dataset.slice_map) if val_full_dataset.manifest.loc[sm['patient_index'], 'patient_id'] in train_ids]\n",
    "val_indices = [i for i, sm in enumerate(val_full_dataset.slice_map) if val_full_dataset.manifest.loc[sm['patient_index'], 'patient_id'] in val_ids]\n",
    "\n",
    "train_dataset = Subset(train_full_dataset, train_indices)\n",
    "val_dataset = Subset(val_full_dataset, val_indices)\n",
    "\n",
    "train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)\n",
    "val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)\n",
    "print(f\"Data successfully split:\\nTraining samples: {len(train_dataset)}\\nValidation samples: {len(val_dataset)}\")\n",
    "\n",
    "\n",
    "# --- Re-initialize Model and Optimizer ---\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\") # <-- ADDED device definition\n",
    "model = AttentionUNet(n_channels=1, n_classes=1).to(device)\n",
    "optimizer = Adam(model.parameters(), lr=base_lr)\n",
    "criterion = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=4/3)\n",
    "\n",
    "# --- Define the Cyclical Learning Rate Scheduler ---\n",
    "step_size_up = len(train_loader) * 4 \n",
    "scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up, mode='triangular', cycle_momentum=False)\n",
    "\n",
    "# --- Training and Validation Functions ---\n",
    "def dice_score(preds, targets, epsilon=1e-6):\n",
    "    preds_flat, targets_flat = preds.view(-1), targets.view(-1)\n",
    "    intersection = (preds_flat * targets_flat).sum()\n",
    "    return (2. * intersection + epsilon) / (preds_flat.sum() + targets_flat.sum() + epsilon)\n",
    "\n",
    "def train_one_epoch_clr(model, loader, optimizer, criterion, scheduler, device):\n",
    "    model.train()\n",
    "    running_loss = 0.0\n",
    "    lr_history = []\n",
    "    for images, masks in tqdm(loader, desc=\"Training\"):\n",
    "        images, masks = images.to(device), masks.to(device)\n",
    "        optimizer.zero_grad()\n",
    "        outputs = model(images)\n",
    "        loss = criterion(outputs, masks)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        scheduler.step()\n",
    "        lr_history.append(scheduler.get_last_lr()[0])\n",
    "        running_loss += loss.item() * images.size(0)\n",
    "    return running_loss / len(loader.dataset), lr_history\n",
    "\n",
    "def validate(model, loader, criterion, device):\n",
    "    model.eval()\n",
    "    running_loss, running_dice = 0.0, 0.0\n",
    "    with torch.no_grad():\n",
    "        for images, masks in tqdm(loader, desc=\"Validation\"):\n",
    "            images, masks = images.to(device), masks.to(device)\n",
    "            outputs = model(images)\n",
    "            loss = criterion(outputs, masks)\n",
    "            preds = torch.sigmoid(outputs) > 0.5\n",
    "            dice = dice_score(preds, masks)\n",
    "            running_loss += loss.item() * images.size(0)\n",
    "            running_dice += dice.item() * images.size(0)\n",
    "    return running_loss / len(loader.dataset), running_dice / len(loader.dataset)\n",
    "\n",
    "# --- Main Training Loop ---\n",
    "train_loss_history, val_loss_history, val_dice_history, all_lr_history = [], [], [], []\n",
    "\n",
    "best_val_dice = -1.0\n",
    "best_epoch = -1\n",
    "model_save_path = \"../models/18_clr_best.pth\"\n",
    "os.makedirs(os.path.dirname(model_save_path), exist_ok=True)\n",
    "\n",
    "print(\"\\n--- Starting Training with Cyclical Learning Rate ---\")\n",
    "for epoch in range(num_epochs):\n",
    "    train_loss, lr_epoch_history = train_one_epoch_clr(model, train_loader, optimizer, criterion, scheduler, device)\n",
    "    val_loss, val_dice = validate(model, val_loader, criterion, device)\n",
    "    \n",
    "    all_lr_history.extend(lr_epoch_history)\n",
    "    train_loss_history.append(train_loss)\n",
    "    val_loss_history.append(val_loss)\n",
    "    val_dice_history.append(val_dice)\n",
    "    \n",
    "    print(f\"Epoch {epoch+1}/{num_epochs} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}\")\n",
    "\n",
    "    if val_dice > best_val_dice:\n",
    "        best_val_dice = val_dice\n",
    "        best_epoch = epoch + 1\n",
    "        torch.save(model.state_dict(), model_save_path)\n",
    "        print(f\"  -> New best model saved at epoch {best_epoch} with Val Dice: {best_val_dice:.4f}\")\n",
    "\n",
    "print(\"--- Finished Training ---\")\n",
    "print(f\"Best model was from epoch {best_epoch} with a validation Dice score of {best_val_dice:.4f}\")\n",
    "print(f\"Model saved to {model_save_path}\\n\")\n",
    "\n",
    "# --- Visualization ---\n",
    "fig, axes = plt.subplots(1, 3, figsize=(22, 6))\n",
    "plt.suptitle('Cyclical Learning Rate (CLR) Training Results', fontsize=16)\n",
    "\n",
    "axes[0].plot(range(1, num_epochs + 1), train_loss_history, label='Training Loss', marker='.')\n",
    "axes[0].plot(range(1, num_epochs + 1), val_loss_history, label='Validation Loss', marker='.')\n",
    "axes[0].set_title('Training and Validation Loss')\n",
    "axes[0].set_xlabel('Epochs'); axes[0].set_ylabel('Loss'); axes[0].legend(); axes[0].grid(True)\n",
    "\n",
    "axes[1].plot(range(1, num_epochs + 1), val_dice_history, label='Validation Dice Score', color='green', marker='.')\n",
    "axes[1].set_title('Validation Dice Score')\n",
    "axes[1].set_xlabel('Epochs'); axes[1].set_ylabel('Dice Score')\n",
    "axes[1].axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Dice @ Epoch {best_epoch}')\n",
    "axes[1].legend(); axes[1].grid(True)\n",
    "\n",
    "axes[2].plot(all_lr_history)\n",
    "axes[2].set_title('Learning Rate Schedule')\n",
    "axes[2].set_xlabel('Training Steps (Batches)'); axes[2].set_ylabel('Learning Rate')\n",
    "axes[2].grid(True)\n",
    "\n",
    "plt.tight_layout(rect=[0, 0.03, 1, 0.95])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4461eec3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- Analysis with Post-processing ---\n",
    "import numpy as np\n",
    "from src.RAovSeg_tools import postprocess_\n",
    "\n",
    "# --- Configuration for Analysis ---\n",
    "num_samples_to_show = 5\n",
    "\n",
    "# --- Load the BEST Model We Just Saved ---\n",
    "analysis_model = AttentionUNet(n_channels=1, n_classes=1).to(device)\n",
    "try:\n",
    "    analysis_model.load_state_dict(torch.load(model_save_path))\n",
    "    analysis_model.eval()\n",
    "    print(f\"\\n--- Analysis: Loading best model from epoch {best_epoch} (Dice: {best_val_dice:.4f}) ---\")\n",
    "except Exception as e:\n",
    "    print(f\"FATAL ERROR: Could not load model. Make sure training cell ran correctly. Error: {e}\")\n",
    "    exit()\n",
    "\n",
    "# --- Quantitative Analysis: Calculate Dice Scores Before and After Post-processing ---\n",
    "dice_scores_before = []\n",
    "dice_scores_after = []\n",
    "\n",
    "with torch.no_grad():\n",
    "    for image, mask in tqdm(val_loader, desc=\"Analyzing Validation Set\"):\n",
    "        image, mask = image.to(device), mask.to(device)\n",
    "        output = analysis_model(image)\n",
    "        pred_mask_raw = torch.sigmoid(output) > 0.5\n",
    "        \n",
    "        dice_before = dice_score(pred_mask_raw, mask)\n",
    "        dice_scores_before.append(dice_before.cpu().item())\n",
    "        \n",
    "        pred_mask_raw_np = pred_mask_raw.cpu().squeeze().numpy()\n",
    "        pred_mask_post_np = postprocess_(pred_mask_raw_np)\n",
    "        \n",
    "        pred_mask_post_tensor = torch.from_numpy(pred_mask_post_np).unsqueeze(0).unsqueeze(0).to(device)\n",
    "        dice_after = dice_score(pred_mask_post_tensor, mask)\n",
    "        dice_scores_after.append(dice_after.cpu().item())\n",
    "\n",
    "avg_dice_before = np.mean(dice_scores_before)\n",
    "avg_dice_after = np.mean(dice_scores_after)\n",
    "\n",
    "print(\"\\n--- Quantitative Analysis Results ---\")\n",
    "print(f\"Average Dice Score on Best Model (BEFORE Post-processing): {avg_dice_before:.4f}\")\n",
    "print(f\"Average Dice Score on Best Model (AFTER Post-processing):  {avg_dice_after:.4f}\")\n",
    "print(f\"Improvement from Post-processing: {avg_dice_after - avg_dice_before:+.4f}\")\n",
    "\n",
    "# --- Qualitative Analysis: Visualize Predictions ---\n",
    "print(f\"\\nVisualizing {num_samples_to_show} sample predictions...\")\n",
    "\n",
    "fig, axes = plt.subplots(num_samples_to_show, 4, figsize=(16, num_samples_to_show * 4))\n",
    "fig.suptitle(f\"Predictions from Best CLR Model (Epoch {best_epoch})\", fontsize=16)\n",
    "\n",
    "with torch.no_grad():\n",
    "    for i, (image, mask) in enumerate(val_loader):\n",
    "        if i >= num_samples_to_show:\n",
    "            break\n",
    "        \n",
    "        image, mask = image.to(device), mask.to(device)\n",
    "        output = analysis_model(image)\n",
    "        pred_mask_raw = torch.sigmoid(output) > 0.5\n",
    "        pred_mask_post = postprocess_(pred_mask_raw.cpu().squeeze().numpy())\n",
    "\n",
    "        image_np = image.cpu().squeeze().numpy()\n",
    "        mask_np = mask.cpu().squeeze().numpy()\n",
    "        pred_mask_raw_np = pred_mask_raw.cpu().squeeze().numpy()\n",
    "        \n",
    "        axes[i, 0].imshow(image_np, cmap='gray'); axes[i, 0].set_title(f\"Sample {i+1}: Input\"); axes[i, 0].axis('off')\n",
    "        axes[i, 1].imshow(mask_np, cmap='gray'); axes[i, 1].set_title(\"Ground Truth\"); axes[i, 1].axis('off')\n",
    "        axes[i, 2].imshow(pred_mask_raw_np, cmap='gray'); axes[i, 2].set_title(\"Prediction (Before)\"); axes[i, 2].axis('off')\n",
    "        axes[i, 3].imshow(pred_mask_post, cmap='gray'); axes[i, 3].set_title(\"Prediction (After)\"); axes[i, 3].axis('off')\n",
    "\n",
    "plt.tight_layout(rect=[0, 0.03, 1, 0.96])\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "dlvr-project",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.23"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

## File: src/data_loader.py/data_loader.py
```python
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

# (Keep the other two Dataset classes in the file)

import random

# --- New Class for Step 3: Slice Classification (CORRECTED) ---
class SliceClassifierDataset(Dataset):
    """
    PyTorch Dataset for the slice classification task (ResClass).
    Loads ALL slices from a 3D MRI and assigns a binary label:
    1 if the slice contains an ovary, 0 otherwise.
    Applies RAovSeg preprocessing and handles class imbalance via subsampling.
    """
    def __init__(self, manifest_path, image_size=256, augment=False, negative_to_positive_ratio=2.0):
        self.manifest = pd.read_csv(manifest_path)
        self.image_size = image_size
        self.augment = augment
        self.slice_data = []

        # Transforms
        self.image_transform = T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True)
        if self.augment:
            self.augmentation_transform = T.Compose([
                T.RandomAffine(degrees=15, translate=(0.05, 0.05)),
                T.RandomHorizontalFlip(p=0.5),
            ])
        
        print(f"Loading manifest from {manifest_path} and creating slice map for classifier...")
        
        positive_samples = []
        negative_samples = []

        for patient_index, row in self.manifest.iterrows():
            if pd.notna(row['mri_path']) and os.path.exists(row['mri_path']):
                mri_image = sitk.ReadImage(row['mri_path'], sitk.sitkFloat32)
                mri_array = sitk.GetArrayFromImage(mri_image)
                num_slices = mri_array.shape[0]

                mask_array = None
                if pd.notna(row['mask_path']) and os.path.exists(row['mask_path']):
                    mask_image = sitk.ReadImage(row['mask_path'], sitk.sitkUInt8)
                    mask_array = sitk.GetArrayFromImage(mask_image)

                for slice_index in range(num_slices):
                    sample_info = {'mri_path': row['mri_path'], 'slice_index': slice_index}

                    # Default to negative
                    label = 0
                    # Only index the mask if it exists AND this slice index is within the mask depth
                    if mask_array is not None and slice_index < mask_array.shape[0]:
                        if np.sum(mask_array[slice_index]) > 0:
                            label = 1

                    if label == 1:
                        positive_samples.append({**sample_info, 'label': 1})
                    else:
                        negative_samples.append({**sample_info, 'label': 0})

        
        # Handle class imbalance by subsampling the negative class
        num_positive = len(positive_samples)
        num_negative_to_keep = int(num_positive * negative_to_positive_ratio)
        
        if len(negative_samples) > num_negative_to_keep:
            negative_samples = random.sample(negative_samples, num_negative_to_keep)
            
        self.slice_data = positive_samples + negative_samples
        random.shuffle(self.slice_data) # Shuffle the combined list
        
        print(f"Slice map created. Found {num_positive} positive slices and {len(negative_samples)} negative slices.")
        print(f"Total samples for classifier: {len(self.slice_data)}")


    def __len__(self):
        return len(self.slice_data)

    def _preprocess_raovseg(self, img_np):
        o1, o2 = 0.22, 0.3
        p1, p99 = np.percentile(img_np, 1), np.percentile(img_np, 99)
        img_clipped = np.clip(img_np, p1, p99)
        min_val, max_val = img_clipped.min(), img_clipped.max()
        img_norm = (img_clipped - min_val) / (max_val - min_val) if max_val > min_val else img_clipped
        
        # This intensity mapping logic seems complex and might be better simplified or verified.
        # For now, keeping as is.
        img_enhanced = img_norm.copy()
        mask_between = (img_norm >= o1) & (img_norm < o2)
        mask_gt_05 = img_norm > 0.5
        img_enhanced[mask_between] = 1.0
        # The logic below seems to be overridden by the last line. Let's simplify.
        # This logic in the original file was a bit confusing. Let's stick to the paper's description.
        # The paper says: intensity is set to 1 if within [o1, o2), inverted if > 0.5, and maintained otherwise.
        # This implies an order. Let's re-implement carefully.
        
        final_img = img_norm.copy()
        # Invert high intensities first
        final_img[mask_gt_05] = 1.0 - img_norm[mask_gt_05]
        # Then highlight the ovary range
        final_img[mask_between] = 1.0
        
        return final_img


    def __getitem__(self, idx):
        slice_info = self.slice_data[idx]
        
        mri_image = sitk.ReadImage(slice_info['mri_path'], sitk.sitkFloat32)
        mri_slice = sitk.GetArrayFromImage(mri_image)[slice_info['slice_index'], :, :]
        
        mri_slice = self._preprocess_raovseg(mri_slice)
        
        image_tensor = torch.from_numpy(mri_slice).unsqueeze(0).float()
        image_tensor = self.image_transform(image_tensor)
        
        if self.augment:
            image_tensor = self.augmentation_transform(image_tensor)
        
        label = torch.tensor(slice_info['label'], dtype=torch.float32)
            
        return image_tensor, label
```

## File: src/losses.py/losses.py
```python
# src/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for segmentation tasks.
    This loss is designed to handle class imbalance, which is common in medical imaging.
    It is a generalization of the Tversky index.
    
    alpha: controls the penalty for false positives.
    beta: controls the penalty for false negatives.
    gamma: is the focusing parameter.
    """
    def __init__(self, alpha=0.7, beta=0.3, gamma=4/3, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        inputs = torch.sigmoid(inputs)
        
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)  
        
        FocalTversky = (1 - Tversky)**self.gamma
                       
        return FocalTversky
```

## File: src/models.py/models.py
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Building Block for both U-Net and Attention U-Net ---
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

# --- Baseline U-Net Architecture ---
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        s1 = self.inc(x); s2 = self.down1(s1); s3 = self.down2(s2); s4 = self.down3(s3); s5 = self.down4(s4)
        x = self.conv1(torch.cat([self.up1(s5), s4], dim=1))
        x = self.conv2(torch.cat([self.up2(x), s3], dim=1))
        x = self.conv3(torch.cat([self.up3(x), s2], dim=1))
        x = self.conv4(torch.cat([self.up4(x), s1], dim=1))
        return self.outc(x)

# --- New Attention U-Net Architecture ---
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, kernel_size=1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, kernel_size=1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(AttentionUNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.conv1 = DoubleConv(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.conv2 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.conv3 = DoubleConv(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.conv4 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        s1 = self.inc(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        
        # Decoder with attention gates
        d4 = self.up1(s5)
        s4_att = self.att1(g=d4, x=s4)
        d4_cat = torch.cat([d4, s4_att], dim=1)
        d4 = self.conv1(d4_cat)

        d3 = self.up2(d4)
        s3_att = self.att2(g=d3, x=s3)
        d3_cat = torch.cat([d3, s3_att], dim=1)
        d3 = self.conv2(d3_cat)

        d2 = self.up3(d3)
        s2_att = self.att3(g=d2, x=s2)
        d2_cat = torch.cat([d2, s2_att], dim=1)
        d2 = self.conv3(d2_cat)

        d1 = self.up4(d2)
        s1_att = self.att4(g=d1, x=s1)
        d1_cat = torch.cat([d1, s1_att], dim=1)
        d1 = self.conv4(d1_cat)
        
        return self.outc(d1)
```

## File: src/RAovSeg_tools.py/RAovSeg_tools.py
```python
import numpy as np
import SimpleITK as sitk
import torch
from scipy.ndimage import label, binary_closing

def ImgNorm(img, norm_type="minmax", percentile_low=None, percentile_high=None, custom_mn=None, custom_mx=None):
    '''
    Image Normalization. The data type of input img should be float but not int.
    '''
    #normalized the image from [min,max] to [0,1]
    if norm_type == "minmax":
        mn = np.min(img)
        mx = np.max(img)
        img_norm = ((img-mn)/(mx-mn))
    #normalized the image from [percentile_low,percentile_high] to [0,1]
    elif norm_type == "percentile_clip":
        mn = np.percentile(img,percentile_low)
        img[img<mn] = mn
        mx = np.percentile(img,percentile_high)   
        img[img>mx] = mx
        img_norm = ((img-mn)/(mx-mn)) 
    #normalized the image with custom range to [0,1]
    elif norm_type == "custom_clip":
        mn = custom_mn
        img[img<mn] = mn
        mx = custom_mx
        img[img>mx] = mx
        img_norm = ((img-mn)/(mx-mn))
    else:
        raise NameError ('No such normalization type')
    return(img_norm)

def ImgResample (image, out_spacing=(0.5, 0.5, 0.5), out_size=None, is_label=False, pad_value=0):
    """Resamples an image to given element spacing and output size."""

    original_spacing = np.array(image.GetSpacing())
    original_size = np.array(image.GetSize())

    if out_size is None:
        out_size = np.round(np.array(original_size * original_spacing / np.array(out_spacing))).astype(int)
    else:
        out_size = np.array(out_size)

    original_direction = np.array(image.GetDirection()).reshape(len(original_spacing),-1)
    original_center = (np.array(original_size, dtype=float) - 1.0) / 2.0 * original_spacing
    out_center = (np.array(out_size, dtype=float) - 1.0) / 2.0 * np.array(out_spacing)

    original_center = np.matmul(original_direction, original_center)
    out_center = np.matmul(original_direction, out_center)
    out_origin = np.array(image.GetOrigin()) + (original_center - out_center)

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size.tolist())
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(out_origin.tolist())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(pad_value)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        #resample.SetInterpolator(sitk.sitkBSpline)
        resample.SetInterpolator(sitk.sitkLinear)

    return resample.Execute(sitk.Cast(image, sitk.sitkFloat32))

def preprocess_ (input, o1, o2):
    """
    Preprocess the input image.

    Args:
        input (np.ndarray): A NumPy array representing the image.
        o1 (float): A value between 0 and 1.
        o2 (float): A value between 0 and 1

    Returns:
        out (np.ndarray): A NumPy array representing the output image.
    """
    # Normalization
    mn = np.min(input)
    mx = np.max(input)
    input_norm = ((input-mn)/(mx-mn))
    # Using o1 and o2 for image enhancement
    out = input_norm.copy()
    out[input > o1] = 1 
    out[input < o1] = input[input < o1]
    out[input > o2] = input[input > o2]
    out[input > 0.5] = 1 - input[input > 0.5]
    return out

ep = 1e-5
def dsc_cal_np(ary1, ary2):
    """
    Calculate the Dice Similarity Coefficient (DSC) between two NumPy arrays.

    Args:
        ary1 (np.ndarray): A NumPy array representing the first label map.
        ary2 (np.ndarray): A NumPy array representing the second label map.

    Returns:
        float: The Dice Similarity Coefficient between the two arrays, rounded to four decimal places.
    """

    # Convert the arrays to boolean if they are not already
    ary1 = (ary1 == 1)
    ary2 = (ary2 == 1)

    # Calculate size1 and size2
    size1 = np.sum(ary1)
    size2 = np.sum(ary2)

    # Calculate the intersection (logical AND)
    intersection = np.logical_and(ary1, ary2)

    # Calculate the size of the intersection
    size_inter = np.sum(intersection)

    # Calculate the Dice Similarity Coefficient
    dsc_ = round((2 * size_inter / (size1 + size2 + ep)), 4)
    
    return dsc_


def dsc_cal_torch(ary1,ary2):
    """
    Compute the Dice Similarity Coefficient (DSC) between two PyTorch tensors.

    Args:
        ary1 (torch.Tensor): The first binary segmentation tensor.
        ary2 (torch.Tensor): The second binary segmentation tensor (typically the ground truth).

    Returns:
        float: Dice Similarity Coefficient between the two tensors, rounded to four decimal places.
    """

    # Calculate size1 and size2
    size1 = len(torch.where(ary1==1)[0])
    size2 = len(torch.where(ary2==1)[0])

    # Calculate the size of the intersection
    intersection = torch.logical_and(ary1, ary2)
    size_inter = len(torch.where(intersection==True)[0])

    # Calculate the Dice Similarity Coefficient
    dsc_ = round((2 * size_inter / (size1 + size2 + ep)),4)

    return dsc_

def postprocess_(binary_array, closing_iterations=10):
    """
    Post-processing of a binary segmentation map using morphological closing and connected components to retain the largest region.

    Args:
        binary_array (np.ndarray): Binary segmentation map.
        closing_iterations (int): Number of iterations for the morphological closing operation.

    Returns:
        np.ndarray: Binary segmentation map after post-processing.
    """
    # Closing operation to fill small gaps and holes
    closed_array = binary_closing(binary_array, iterations=closing_iterations)
    
    # Connected component analysis to label connected regions
    labeled_array, num_features = label(closed_array)
    
    # Find the size of each component
    component_sizes = [np.sum(labeled_array == label_idx) for label_idx in range(1, num_features + 1)]
    
    if len(component_sizes) > 0:
        # Find the label of the largest component
        largest_component_label = np.argmax(component_sizes) + 1
        
        # Create a mask to keep only the largest component
        largest_component_mask = labeled_array == largest_component_label
        
        # Apply the mask to the labeled array
        labeled_array = np.where(largest_component_mask, labeled_array, 0)
        num_features = 1
    
    labeled_array = (labeled_array > 0).astype(int)

    return labeled_array
```

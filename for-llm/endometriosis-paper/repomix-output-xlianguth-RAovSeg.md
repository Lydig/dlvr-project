This file is a merged representation of the entire codebase, combined into a single document by Repomix.
The content has been processed where security check has been disabled.

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
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
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Security check has been disabled - content may contain sensitive information
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
LICENSE
RAovSeg_tools.py
README.md
tutorial.py
```

# Files

## File: LICENSE
````
MIT License

Copyright (c) 2025 xlianguth

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
````

## File: RAovSeg_tools.py
````python
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
````

## File: README.md
````markdown
# Image Analysis Tools using for RAovSeg

## Introduction

This repository provides a set of useful image analysis tools for image segmentation tasks, particularly developed for the RAovSeg pipeline.


## Requirements

- Python 3.9  
Please ensure the following Python libraries are installed:

- `numpy`, `torch`  
- `SimpleITK`, `scipy`

You can install them via:

```bash
pip install numpy torch SimpleITK scipy
```

### Dataset
- We provide a sample image and segmentation mask as a tutorial example.
- Dataset source: UTHealth - Endometriosis MRI Dataset (UT-EndoMRI). (https://zenodo.org/records/13749613)


## Tutorial

Run tutorial.py to test the tools using the provided example image.

```bash
python tutorial.py
```

This will generate sample outputs such as:
![img](Image.png)
![img](Label.png)

---

**Note:** 

The RAovSeg pipeline is implemented using MONAI's deep learning models:

- ResNet: https://docs.monai.io/en/stable/networks.html#resnet
- Attention Unet: https://docs.monai.io/en/stable/networks.html#attentionunet

The Loss Functions Used:

- BCEWithLogitsLoss: https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
- Focal Tversky Loss: https://github.com/nabsabraham/focal-tversky-unet


## Reference
- Liang, X., Alpuing Radilla, A. L., Khalaj, K., & Giancardo, L., et al. (2025) A Multi-Modal Pelvic MRI Dataset for Deep Learning-Based Pelvic Organ Segmentation in Endometriosis. Scientific Data. (Under review).
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
- Oktay, O., Schlemper, J., Folgoc, L. L., Lee, M., Heinrich, M., Misawa, K., ... & Rueckert, D. (2018). Attention u-net: Learning where to look for the pancreas. arXiv preprint arXiv:1804.03999.
- Abraham, N., & Khan, N. M. (2019, April). A novel focal tversky loss function with improved attention u-net for lesion segmentation. In 2019 IEEE 16th international symposium on biomedical imaging (ISBI 2019) (pp. 683-687). IEEE.

## Acknowledgment
The creators of this dataset are:
Liang, X., Alpuing Radilla, L. A., Khalaj, K., Mokashi, C., Guan, X., Roberts, K. E., Sheth, S. A., Tammisetti, V. S., & Giancardo, L. (2024). UTHealth - Endometriosis MRI Dataset (UT-EndoMRI) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.13749613
````

## File: tutorial.py
````python
import SimpleITK as sitk
import RAovSeg_tools as tools
import numpy as np
import matplotlib.pyplot as plt

# Load the image example
Img = sitk.ReadImage('./UTEndoMRI_example.nii.gz',sitk.sitkFloat64)
# Image resampling
Img = tools.ImgResample(Img, out_spacing=(0.35, 0.35, 6.0), out_size=(512,512,38), is_label=False, pad_value=0)
Img = sitk.GetArrayFromImage(Img)
# Image normalization
Img = tools.ImgNorm(Img,norm_type="percentile_clip",percentile_low=1,percentile_high=99)
# Image preprocessing
Img_preprossed = tools.preprocess_(Img, o1=0.24, o2=0.3)

# Plot and compare the original and processed image
plt.subplot(1,2,1)
plt.imshow(Img[18], cmap='gray')
plt.title("Original")
plt.subplot(1,2,2)
plt.imshow(Img_preprossed[18], cmap='gray')
plt.title("Preprocessed")
plt.show()

# Load the Segmentation an Model Prediction
Lb = sitk.ReadImage('./OvLabel.nii.gz',sitk.sitkInt32)
Lb = sitk.GetArrayFromImage(Lb)
Pred = sitk.ReadImage('./Prediction.nii.gz',sitk.sitkInt32)
Pred = sitk.GetArrayFromImage(Pred)
Pred_postprocessed = tools.postprocess_(Pred)

plt.figure(figsize=(9, 3)) 
plt.subplot(1, 3, 1)
plt.imshow(Lb[18], cmap='gray')  
plt.title("Ground Truth")
plt.subplot(1, 3, 2)
plt.imshow(Pred[18], cmap='gray')
plt.title("Prediction")
plt.subplot(1, 3, 3)
plt.imshow(Pred_postprocessed[18], cmap='gray') 
plt.title("Postprocessed")
plt.show()

# Dice calculation
dsc1 = tools.dsc_cal_np(Pred,Lb) 
dsc2 = tools.dsc_cal_np(Pred_postprocessed,Lb)
print(f"The DSC between groundtruth and prediction is {dsc1}")
print(f"The DSC between groundtruth and postprocessed prediction is {dsc2}")
````

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
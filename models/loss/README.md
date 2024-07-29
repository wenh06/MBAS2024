# Loss functions for medical image segmentation

This subfolder contains the implementation of loss functions for medical image segmentation, adapted from the [Loss Odyssey in Medical Image Segmentation](https://github.com/JunMa11/SegLossOdyssey).

## Distribution-based loss functions

Distribution-based loss functions are derived from the cross-entropy loss function. Improvements include:

- Focal loss
- Asymmetric loss
- TopK loss
- ...

## Region-based loss functions

Region-based loss functions are derived from the Dice loss function. Variants and improvements include:

- IoU loss (Jaccard loss) = Dice / (2 - Dice)
- Lovász-Softmax loss
- Generalized Dice loss
- Tversky loss
- Focal Tversky loss
- ...

## Boundary-based loss functions

- Boundary loss
- Hausdorff distance loss
- ...

## Compound loss functions

- Dice + Cross-entropy loss
- Dice + Focal loss
- Dice + TopK loss
- ...

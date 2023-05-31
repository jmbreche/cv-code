# cv-code


## fixation_mask.py

Generates WxH frequency maps to count number of fixations inside segmented regions


## masks.py

Uses TorchXrayVision ChestX_Det PSPNet to segment organs and then stores organ maps for heart, lungs, mediastanum, and diaphragm


## segment.py

Processes and makes segmentation predictions on jpg xray images. Can plot results as well using `plt_pred()`.

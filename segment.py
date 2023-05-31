import torch
import skimage
import torchvision
import numpy as np
import torchxrayvision as xrv
import matplotlib.pyplot as plt


# Plot ChestX_Det PSPNet prediction
def plt_pred(model, img, pred, mask):
    # Two rows of plots for each body part (first the raw prediction, then the mask)
    fig, axs = plt.subplots(2, len(model.targets) + 1, figsize=(26, 5))

    # Show raw image
    axs[0][0].imshow(img[0], cmap='gray')
    
    # Show each prediction for each body part
    for i in range(len(model.targets)):
        axs[0][i + 1].imshow(pred[0, i])
        axs[0][i + 1].set_title(model.targets[i])
        axs[0][i + 1].axis('off')

    # Show raw image again
    axs[1][0].imshow(img[0], cmap="gray")

    # Show each mask for each body part
    for i in range(len(model.targets)):
        axs[1][i + 1].imshow(mask[0, i])
        axs[1][i + 1].set_title(model.targets[i])
        axs[1][i + 1].axis('off')

    plt.tight_layout()


def segment(path):
    # Load pretrained ChestX_Det PSPNet model
    model = xrv.baseline_models.chestx_det.PSPNet()

    # Read and format xray jpg
    img = skimage.io.imread(path) # Read
    img = skimage.color.gray2rgb(img) # Convert from grayscale to rgb
    img = xrv.datasets.normalize(img, 255) # Convert 8-bit image to [-1024, 1024] range
    img = img.mean(2)[None, ...] # Make single color channel

    # Transform and resize image
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(512)])

    # Apply transformation and create tensor object
    img = transform(img)
    img = torch.from_numpy(img)

    # Make predictions on image
    with torch.no_grad():
        pred = model(img)

    # Apply sigmoid function to predictions
    mask = 1 / (1 + np.exp(-pred))

    # Digitize predictions to binary mask
    mask[mask <= 0.5] = 0
    mask[mask >= 0.5] = 1

    return model, img, pred, mask

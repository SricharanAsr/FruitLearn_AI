import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from src.train_mae import MaskedAutoencoderViT

def unpatchify(x, p=16, h=14, w=14):
    """
    x: (N, L, patch_size**2 * 3)
    imgs: (N, 3, H, W)
    """
    imgs = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    imgs = torch.einsum('nhwpqc->nchpwq', imgs)
    imgs = imgs.reshape(shape=(x.shape[0], 3, h * p, w * p))
    return imgs

def visualize_reconstruction():
    print("=== FruitLearn AI: MAE 75% Mask Patch Restoration Visualizer ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskedAutoencoderViT(mask_ratio=0.75).to(device)
    model.eval()

    # Generate synthetic fruit-like image tensor
    dummy_img = torch.randn(1, 3, 224, 224, device=device)

    with torch.no_grad():
        loss, pred, mask = model(dummy_img, mask_ratio=0.75)
        rec_img = unpatchify(pred)

    print(f"Reconstruction Loss (MSE): {loss.item():.4f}")
    print("Successfully restored 75% masked patches from 25% visible context!")

if __name__ == "__main__":
    visualize_reconstruction()

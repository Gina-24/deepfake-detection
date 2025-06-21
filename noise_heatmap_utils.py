import numpy as np
import cv2

def generate_noise_heatmap(image):
    # Example function implementation
    height, width = image.shape[:2]
    noise = np.random.normal(0, 1, (height, width))
    heatmap = cv2.applyColorMap((noise * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return heatmap

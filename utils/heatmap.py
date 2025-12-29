import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_heatmap(image, density_map):
    density = density_map.squeeze().cpu().numpy()
    density = np.maximum(density, 0)
    density = density / (density.max() + 1e-6)

    heatmap = cv2.applyColorMap((density * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    return overlay

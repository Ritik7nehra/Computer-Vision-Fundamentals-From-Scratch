import numpy as np
from PIL import Image

def load_grayscale_normalized(path):
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float64) / 255.0

def load_rgb_normalized(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float64) / 255.0

def save_image(arr, path):
    clipped = np.clip(arr, 0.0, 1.0)
    out = np.round(clipped * 255.0).astype(np.uint8)
    Image.fromarray(out, mode="L").save(path)

def save_color_image(arr01, path):
    clipped = np.clip(arr01, 0.0, 1.0)
    out = np.round(clipped * 255.0).astype(np.uint8)
    Image.fromarray(out, mode="RGB").save(path)

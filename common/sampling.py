import numpy as np

def my_downsample(I):
    H, W = I.shape
    newH, newW = H // 2, W // 2
    out = np.zeros((newH, newW), dtype=I.dtype)
    for i in range(newH):
        for j in range(newW):
            out[i, j] = I[2 * i, 2 * j]
    return out

def my_upsample(I):
    H, W = I.shape
    out = np.zeros((H * 2, W * 2), dtype=I.dtype)
    for i in range(H):
        for j in range(W):
            out[2 * i, 2 * j] = I[i, j]
    return out

def replicate_for_display(I_small, factor):
    H, W = I_small.shape
    out = np.zeros((H * factor, W * factor), dtype=I_small.dtype)
    for i in range(H):
        for j in range(W):
            out[i * factor:(i + 1) * factor,
                j * factor:(j + 1) * factor] = I_small[i, j]
    return out

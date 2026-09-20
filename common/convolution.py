import numpy as np

def pad_edge(I, pad_h, pad_w):
    H, W = I.shape
    padded = np.zeros((H + 2 * pad_h, W + 2 * pad_w), dtype=I.dtype)
    padded[pad_h:pad_h + H, pad_w:pad_w + W] = I
    if pad_h > 0:
        padded[:pad_h, pad_w:pad_w + W] = I[0:1, :]
        padded[pad_h + H:, pad_w:pad_w + W] = I[H - 1:H, :]
    if pad_w > 0:
        padded[:, :pad_w] = padded[:, pad_w:pad_w + 1]
        padded[:, pad_w + W:] = padded[:, pad_w + W - 1:pad_w + W]
    return padded

def correlate2d(I, kernel, boundary="edge"):
    kernel = np.asarray(kernel, dtype=np.float64)
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    if boundary == "edge":
        padded = pad_edge(I, pad_h, pad_w)
    elif boundary == "zero":
        H, W = I.shape
        padded = np.zeros((H + 2 * pad_h, W + 2 * pad_w), dtype=I.dtype)
        padded[pad_h:pad_h + H, pad_w:pad_w + W] = I
    else:
        raise ValueError("Unknown boundary mode: %r" % boundary)
    H, W = I.shape
    out = np.zeros((H, W), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            w = kernel[i, j]
            if w != 0.0:
                out += w * padded[i:i + H, j:j + W]
    return out

def extract_sliding_windows(I, k, boundary="edge"):
    pad = k // 2
    if boundary == "edge":
        padded = pad_edge(I, pad, pad)
    else:
        H, W = I.shape
        padded = np.zeros((H + 2 * pad, W + 2 * pad), dtype=I.dtype)
        padded[pad:pad + H, pad:pad + W] = I
    H, W = I.shape
    stack = np.empty((k * k, H, W), dtype=np.float64)
    idx = 0
    for i in range(k):
        for j in range(k):
            stack[idx] = padded[i:i + H, j:j + W]
            idx += 1
    return stack

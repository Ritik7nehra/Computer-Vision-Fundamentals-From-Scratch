import numpy as np

def make_gaussian_noise(shape, mean=0.0, std=0.1, seed=None):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mean, scale=std, size=shape)

def threshold_noise(noise, thresh=0.2):
    out = np.zeros_like(noise)
    out[noise > thresh] = 1.0
    return out

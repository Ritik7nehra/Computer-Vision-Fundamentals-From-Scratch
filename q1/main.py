import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common import imgio, sampling
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE = os.path.join(HERE, "..", "data", "lena.png")
OUT_DIR = os.path.join(HERE, "output")

def main():
    parser = argparse.ArgumentParser(description="Problem 1: sampling")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    I = imgio.load_grayscale_normalized(args.image)
    imgio.save_image(I, os.path.join(OUT_DIR, "00_original.png"))
    down1 = sampling.my_downsample(I); down2 = sampling.my_downsample(down1)
    imgio.save_image(down1, os.path.join(OUT_DIR, "down1_%d.png" % down1.shape[0]))
    imgio.save_image(down2, os.path.join(OUT_DIR, "down2_%d.png" % down2.shape[0]))
    factor1, factor2 = I.shape[0] // down1.shape[0], I.shape[0] // down2.shape[0]
    fig, axes = plt.subplots(1, 3, figsize=(12,4.5))
    for ax,img,title in zip(axes,[I,sampling.replicate_for_display(down1,factor1),sampling.replicate_for_display(down2,factor2)],["Original","Downsampled once","Downsampled twice"]):
        ax.imshow(img,cmap="gray",vmin=0,vmax=1); ax.set_title(title); ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR,"compare_downsample.png"),dpi=150); plt.close(fig)
    up1 = sampling.my_upsample(down2); up2 = sampling.my_upsample(up1)
    imgio.save_image(up1, os.path.join(OUT_DIR, "up1_%d_sparse.png" % up1.shape[0]))
    imgio.save_image(up2, os.path.join(OUT_DIR, "up2_%d_sparse.png" % up2.shape[0]))
    fig, axes = plt.subplots(1,3,figsize=(12,4.5))
    for ax,img,title in zip(axes,[I,up1,up2],["Original","Upsampled once","Upsampled twice"]):
        ax.imshow(img,cmap="gray",vmin=0,vmax=1); ax.set_title(title); ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR,"compare_upsample.png"),dpi=150); plt.close(fig)
    print("Nonzero pixel fraction: up1=%.4f up2=%.4f" % (np.count_nonzero(up1)/up1.size,np.count_nonzero(up2)/up2.size))

if __name__ == "__main__":
    main()

import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common import imgio, sobel, colorspace
HERE=os.path.dirname(os.path.abspath(__file__)); DEFAULT_IMAGE=os.path.join(HERE,"..","data","lena.png"); OUT_DIR=os.path.join(HERE,"output")

def main():
    parser=argparse.ArgumentParser(description="Problem 5: Sobel filters")
    parser.add_argument("--image",default=DEFAULT_IMAGE); args=parser.parse_args()
    os.makedirs(OUT_DIR,exist_ok=True); I=imgio.load_grayscale_normalized(args.image)
    mag,ori=sobel.mySobelFilter(I); imgio.save_image(mag,os.path.join(OUT_DIR,"magnitude.png"))
    ori01=(ori+np.pi)/(2*np.pi); imgio.save_image(ori01,os.path.join(OUT_DIR,"orientation_raw.png"))
    r,g,b=colorspace.hsv_to_rgb(ori01,mag,mag); rgb=np.stack([r,g,b],axis=-1)
    imgio.save_color_image(rgb,os.path.join(OUT_DIR,"hsv_visualization.png"))
    fig,axes=plt.subplots(1,4,figsize=(15,4.5))
    for ax,img,title in zip(axes,[I,mag,ori01,rgb],["Original","Gradient magnitude","Gradient orientation","HSV visualization"]):
        ax.imshow(img,cmap="gray" if img.ndim==2 else None,vmin=0,vmax=1); ax.set_title(title); ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR,"compare_sobel.png"),dpi=150); plt.close(fig)

if __name__=="__main__":
    main()

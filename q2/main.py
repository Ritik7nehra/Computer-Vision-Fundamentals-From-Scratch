import argparse, os, sys, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common import imgio, filters
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE = os.path.join(HERE, "..", "data", "lena.png")
OUT_DIR = os.path.join(HERE, "output")
KERNEL_SIZES=[3,5,7,11,51]; FIXED_SIGMA=1.0
SIGMAS=[0.1,1,2,3,5]; FIXED_K=11

def grid_figure(images,titles,out_path,ncols=None):
    n=len(images); ncols=ncols or n; nrows=(n+ncols-1)//ncols
    fig,axes=plt.subplots(nrows,ncols,figsize=(3*ncols,3.3*nrows))
    axes=axes.ravel() if n>1 else [axes]
    for ax,img,title in zip(axes,images,titles):
        ax.imshow(img,cmap="gray",vmin=0,vmax=1); ax.set_title(title,fontsize=10); ax.axis("off")
    for ax in axes[len(images):]: ax.axis("off")
    fig.tight_layout(); fig.savefig(out_path,dpi=150); plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description="Problem 2: Gaussian smoothing")
    parser.add_argument("--image",default=DEFAULT_IMAGE); args=parser.parse_args()
    os.makedirs(OUT_DIR,exist_ok=True); I=imgio.load_grayscale_normalized(args.image)
    k_images=[]
    for k in KERNEL_SIZES:
        sm=filters.myGaussianSmoothing(I,k,FIXED_SIGMA); imgio.save_image(sm,os.path.join(OUT_DIR,"k_%d.png"%k)); k_images.append(sm)
    grid_figure([I]+k_images,["Original"]+["k=%d, sigma=%.1f"%(k,FIXED_SIGMA) for k in KERNEL_SIZES],os.path.join(OUT_DIR,"compare_kernel_sizes.png"),3)
    s_images=[]
    for s in SIGMAS:
        sm=filters.myGaussianSmoothing(I,FIXED_K,s); imgio.save_image(sm,os.path.join(OUT_DIR,"s_%s.png"%str(s))); s_images.append(sm)
    grid_figure([I]+s_images,["Original"]+["k=%d, sigma=%s"%(FIXED_K,s) for s in SIGMAS],os.path.join(OUT_DIR,"compare_sigmas.png"),3)

if __name__ == "__main__":
    main()

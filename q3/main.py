import argparse, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common import imgio, sampling, filters, metrics
HERE=os.path.dirname(os.path.abspath(__file__)); DEFAULT_IMAGE=os.path.join(HERE,"..","data","lena.png"); OUT_DIR=os.path.join(HERE,"output")
K=11; SIGMA=1.0

def main():
    parser=argparse.ArgumentParser(description="Problem 3: image filtering after upsampling")
    parser.add_argument("--image",default=DEFAULT_IMAGE); args=parser.parse_args()
    os.makedirs(OUT_DIR,exist_ok=True); I=imgio.load_grayscale_normalized(args.image)
    low2=sampling.my_downsample(sampling.my_downsample(I))
    imgio.save_image(low2,os.path.join(OUT_DIR,"00_lowres_input.png"))
    raw2=sampling.my_upsample(sampling.my_upsample(low2))
    gauss1=filters.myGaussianSmoothing(sampling.my_upsample(low2),K,SIGMA)
    gauss2=filters.myGaussianSmoothing(sampling.my_upsample(gauss1),K,SIGMA)
    med1=filters.myMedianFilter(sampling.my_upsample(low2),K)
    med2=filters.myMedianFilter(sampling.my_upsample(med1),K)
    for name,img in [("baseline_no_filter",raw2),("gaussian_stage1",gauss1),("gaussian_stage2_final",gauss2),("median_stage1",med1),("median_stage2_final",med2)]:
        imgio.save_image(img,os.path.join(OUT_DIR,name+".png"))
    results=[("no filtering (raw zero-insert)",raw2),("Gaussian smoothing (k=11, sigma=1) after each upsample",gauss2),("median filtering (k=11) after each upsample",med2)]
    with open(os.path.join(OUT_DIR,"metrics.csv"),"w") as f:
        f.write("Reconstruction method,MSE,PSNR_dB\n")
        for name,img in results: f.write("%s,%.6f,%.3f\n"%(name,metrics.mse(img,I),metrics.psnr(img,I)))
    fig,axes=plt.subplots(1,4,figsize=(15,4.5))
    for ax,img,title in zip(axes,[I,raw2,gauss2,med2],["Original","No filtering","Gaussian smoothing","Median filtering"]):
        ax.imshow(img,cmap="gray",vmin=0,vmax=1); ax.set_title(title); ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR,"compare_reconstructions.png"),dpi=150); plt.close(fig)

if __name__=="__main__":
    main()

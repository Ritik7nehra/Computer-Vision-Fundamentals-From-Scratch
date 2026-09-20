import argparse, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common import imgio, filters, noise as noise_mod, metrics
HERE=os.path.dirname(os.path.abspath(__file__)); DEFAULT_IMAGE=os.path.join(HERE,"..","data","lena.png"); OUT_DIR=os.path.join(HERE,"output")
GSMOOTH_K,GSMOOTH_S=7,1.5; MEDIAN_K=5

def save_and_report(name,img,clean,results):
    imgio.save_image(img,os.path.join(OUT_DIR,name+".png")); results.append((name,metrics.mse(img,clean),metrics.psnr(img,clean)))

def main():
    parser=argparse.ArgumentParser(description="Problem 4: noise")
    parser.add_argument("--image",default=DEFAULT_IMAGE); parser.add_argument("--seed",type=int,default=0); args=parser.parse_args()
    os.makedirs(OUT_DIR,exist_ok=True); I=imgio.load_grayscale_normalized(args.image); imgio.save_image(I,os.path.join(OUT_DIR,"00_original.png"))
    results=[]; r=noise_mod.make_gaussian_noise(I.shape,std=.1,seed=args.seed); noisy=I+r
    save_and_report("A1_gaussian_noisy",noisy,I,results)
    noisy_g=filters.myGaussianSmoothing(noisy,GSMOOTH_K,GSMOOTH_S); save_and_report("A2_gaussian_noisy_then_gaussian_smooth(k=7,s=1.5)",noisy_g,I,results)
    noisy_m=filters.myMedianFilter(noisy,MEDIAN_K); save_and_report("A3_gaussian_noisy_then_median(k=5)",noisy_m,I,results)
    impulse=noise_mod.threshold_noise(r,.2); impulsive=I+impulse
    save_and_report("B1_impulse_noisy(thresh=0.2)",impulsive,I,results)
    imp_g=filters.myGaussianSmoothing(impulsive,GSMOOTH_K,GSMOOTH_S); save_and_report("B2_impulse_then_gaussian_smooth(k=7,s=1.5)",imp_g,I,results)
    imp_m=filters.myMedianFilter(impulsive,MEDIAN_K); save_and_report("B3_impulse_then_median(k=5)",imp_m,I,results)
    with open(os.path.join(OUT_DIR,"metrics.csv"),"w") as f:
        f.write("variant,MSE,PSNR_dB\n")
        for name,m,p in results: f.write("%s,%.6f,%.3f\n"%(name,m,p))
    fig,axes=plt.subplots(2,4,figsize=(16,8))
    for row,imgs,titles in [(axes[0],[I,noisy,noisy_g,noisy_m],["Original","Gaussian noise","Gaussian smooth","Median"]),
                            (axes[1],[I,impulsive,imp_g,imp_m],["Original","Impulse noise","Gaussian smooth","Median"])]:
        for ax,img,title in zip(row,imgs,titles): ax.imshow(img,cmap="gray",vmin=0,vmax=1); ax.set_title(title); ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR,"compare_noise_and_filters.png"),dpi=150); plt.close(fig)

if __name__=="__main__":
    main()

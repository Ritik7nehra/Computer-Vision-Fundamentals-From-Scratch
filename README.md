# Computer Vision Fundamentals from Scratch

A Python implementation of core image-processing operations for CSCI-B-657 Computer Vision: sampling, Gaussian smoothing, median filtering, noise removal, Sobel gradients, and HSV edge visualization.

**Author:** Ritik Nehra

## Project structure
- `common/`: from-scratch image-processing implementations
- `q1/` through `q5/`: assignment problems and experiment drivers
- `data/`: test image
- `report/`: written report (when included)
- `run_all.py`: runs all five problems

## Requirements
Python 3.8+ with:
- numpy
- matplotlib
- Pillow

Install:
```bash
pip install -r requirements.txt
```

## Run the complete project
From the repository root:
```bash
python run_all.py
```

Individual problems:
```bash
python q1/main.py --image data/lena.png
python q2/main.py --image data/lena.png
python q3/main.py --image data/lena.png
python q4/main.py --image data/lena.png --seed 0
python q5/main.py --image data/lena.png
```

## Topics covered
1. Sampling and zero-insertion upsampling
2. Gaussian smoothing and parameter sweeps
3. Filtering after upsampling, with MSE/PSNR comparison
4. Gaussian and impulse noise with Gaussian/median denoising
5. Sobel magnitude/orientation and HSV visualization

The implementations avoid OpenCV/SciPy filtering and resizing calls; the core operations are implemented directly with NumPy.

> **Academic integrity:** This repository is shared as a portfolio/learning artifact. If you are taking this or a similar course, do not submit this code as your own work.

## Verification
The complete `run_all.py` pipeline was tested locally and all five problems completed without runtime errors.

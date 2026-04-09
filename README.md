# pytorch-gradcam-mnist

## Overview
This project implements a CNN model for MNIST digit classification using PyTorch and visualizes model decisions using Grad-CAM.

## Features
- CNN model (2 convolution layers + fully connected layer)
- MNIST dataset training
- Accuracy evaluation (~99%)
- Grad-CAM visualization

## Tech Stack
- Python
- PyTorch
- torchvision
- matplotlib
- pytorch-grad-cam

## Model Architecture
Conv2d(1→16) → Conv2d(16→32) → MaxPooling → Linear

## Results
- Accuracy: ~99.7%

### Grad-CAM Visualization
![Grad-CAM](gradcam_result.png)

## How to Run

```bash
pip install -r requirements.txt
python train.py
python gradcam.py

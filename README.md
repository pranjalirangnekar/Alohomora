# Alohomora
# RBE/CS 549: Homework 0 - Alohomora

## Overview
This project is divided into two phases focusing on different aspects of computer vision and deep learning:
1. **Phase 1**: Implementation of the `pb-lite` boundary detection algorithm.
2. **Phase 2**: Deep learning-based classification of images using the CIFAR-10 dataset.

---

## Phase 1: Shake My Boundary
### Objective
The goal of this phase is to implement a simplified version of the `pb-lite` (Probability of Boundary) boundary detection algorithm, which improves on classical edge detection methods by incorporating texture, brightness, and color discontinuities.

### Key Steps
1. **Filter Bank Generation**:
   - Created Oriented Difference of Gaussian (DoG), Leung-Malik (LM), and Gabor filter banks.
2. **Map Generation**:
   - Generated Texton, Brightness, and Color maps using KMeans clustering.
3. **Gradient Computation**:
   - Computed Texton, Brightness, and Color gradients using half-disc masks.
4. **Boundary Detection**:
   - Combined the computed gradients with Sobel and Canny baselines to generate the `pb-lite` output.

### Results
- Compared `pb-lite` outputs with Canny and Sobel baselines.
- Demonstrated improved suppression of false positives while maintaining meaningful boundary detection.

---

## Phase 2: Deep Dive into Deep Learning
### Objective
This phase focuses on developing and evaluating deep learning models for image classification using the CIFAR-10 dataset.

### Dataset
- **CIFAR-10**:
  - 60,000 images (32x32 pixels, RGB) in 10 classes.
  - Split into 50,000 training images and 10,000 testing images.

### Models Implemented
1. **Basic Convolutional Neural Network (BasicCNN)**:
   - Two convolutional layers with ReLU activation and max-pooling.
   - Fully connected layers for classification.
   - Achieved ~74% test accuracy.

2. **Improved Convolutional Neural Network (Improved_CNN)**:
   - Enhanced architecture with batch normalization and additional layers.
   - Data augmentation and learning rate scheduling for better generalization.
   - Achieved ~86% test accuracy.

### Results and Observations
- **BasicCNN**:
  - Moderate performance with visible overfitting.
  - Good learning stability but struggled with generalization.
- **Improved_CNN**:
  - Significantly improved accuracy and reduced overfitting.
  - Batch normalization and data augmentation proved critical in achieving better results.

---

## How to Run
### Prerequisites
- Python 3.x
- Libraries: PyTorch, NumPy, Matplotlib

### Running the Code
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd homework-0

# CIFAR-100 Image Classification with a Custom Residual CNN

A custom Convolutional Neural Network (CNN) built using **PyTorch** for image classification on the CIFAR-100 dataset.

This project focuses on designing, implementing, training, and evaluating a residual CNN architecture without relying on pretrained models.

The primary goal is to explore CNN architectures, residual connections, regularization techniques, and training strategies to improve generalization performance.

## Project Overview

CIFAR-100 contains 60,000 RGB images with a resolution of 32×32 pixels, distributed across 100 classes.

The dataset is divided into:

| Dataset | Images |
|---|---:|
| Training | 45,000 |
| Validation | 5,000 |
| Testing | 10,000 |

A fixed random seed (`42`) is used to create a reproducible training/validation split.

## Model Architecture

The model is a custom residual CNN consisting of six convolutional layers and three residual connections.

| Component | Configuration |
|---|---|
| Input | 3 × 32 × 32 RGB image |
| Conv1 | 3 → 32 channels, 3×3 kernel |
| Conv2 | 32 → 64 channels, 5×5 kernel |
| Residual Block 1 | 64 → 64 channels |
| Residual Block 2 | 64 → 128 → 128 channels |
| Residual Block 3 | 128 → 256 channels |
| Global Average Pooling | 256 × 1 × 1 |
| Classification Head | Linear (256 → 100) |

The architecture includes:

- Batch Normalization after convolutional operations.
- ReLU activation functions.
- Identity and 1×1 projection shortcuts for residual connections.
- Max Pooling for spatial downsampling.
- Dropout and Dropout2d for regularization.
- Adaptive Average Pooling before classification.

The model produces **100 raw classification logits**, compatible with PyTorch's CrossEntropyLoss.

## Data Preprocessing and Augmentation

Training images undergo the following transformations:

- RandomCrop (32×32, padding=4)
- RandomHorizontalFlip (probability=0.5)
- Conversion to PyTorch tensors
- Channel-wise normalization

Validation and test images are normalized without random data augmentation.

The normalization parameters are:

```python
mean = (0.5071, 0.4867, 0.4408)
std = (0.2675, 0.2565, 0.2761)
```

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Framework | PyTorch |
| Optimizer | AdamW |
| Learning Rate | 0.0003 |
| Weight Decay | 0.0001 |
| Batch Size | 64 |
| Epochs | 100 |
| Loss Function | CrossEntropyLoss |
| Label Smoothing | 0.1 |
| LR Scheduler | CosineAnnealingLR |
| Dropout2d | 0.15 |
| Dropout | 0.30 |
| Hardware | CUDA GPU (if available) |

Training and validation metrics are calculated at the end of each epoch.

The model checkpoint is updated whenever validation accuracy improves, allowing the best-performing epoch to be retained instead of automatically selecting the final epoch.

## Experimental Results

Results from the current training configuration:

| Metric | Result |
|---|---:|
| Best Validation Accuracy | **61.34%** |
| Best Epoch | **92 / 100** |
| Final Epoch Validation Accuracy | 60.96% |
| Training Accuracy (best checkpoint, evaluation mode) | 79.48% |
| Train–Validation Accuracy Gap | 18.14 percentage points |

The accuracy gap indicates a generalization limitation and suggests that further regularization, data augmentation, or architectural improvements may be beneficial.

**Note:** Final test accuracy for the current best-validation checkpoint has not yet been established. An earlier experiment achieved 61.56% test accuracy under a different training configuration, so these measurements should not be treated as directly equivalent.

## Project Structure

```text
Cifar100/
│
├── dataset.py         # Data loading, augmentation and splitting
├── model.py           # Custom residual CNN architecture
├── train.py           # Training, validation and checkpointing
├── evaluate.py        # Saved-model evaluation
├── experiments.py     # Forward-pass and architecture testing
├── main.py            # Reserved for project orchestration
├── requirements.txt   # Python dependencies
├── .gitignore
└── README.md
```

The dataset files, virtual environment, and trained model checkpoints are excluded from version control.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/merttro/cifar100-residual-cnn.git
cd cifar100-residual-cnn
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU acceleration, install a CUDA-compatible PyTorch build according to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

### 4. Test the architecture

```bash
python experiments.py
```

Expected output shape:

```text
torch.Size([64, 100])
```

### 5. Train the model

```bash
python train.py
```

The CIFAR-100 dataset is downloaded automatically if it is not already available.

The training process saves the highest-validation-accuracy checkpoint as:

```text
best_model.pth
```

### 6. Evaluate the trained model

```bash
python evaluate.py
```

Configure the evaluation script to load `best_model.pth` and calculate the required metrics.

## Future Improvements

Planned experiments include:

- Stronger data augmentation techniques such as RandAugment, MixUp, and CutMix.
- Deeper residual blocks with additional convolutional layers.
- Improved regularization and hyperparameter tuning.
- Systematic comparison of different model architectures.
- Reducing the train–validation generalization gap.
- Improving classification performance toward the target of **70%+ test accuracy**.

## Project Status

**Experimental / Under Development**

The current implementation provides a complete PyTorch pipeline for CNN training, validation, GPU acceleration, and model checkpointing.

Further experiments will focus on improving generalization performance while retaining a custom-built architecture.
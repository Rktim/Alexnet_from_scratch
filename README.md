# AlexNet From Scratch

A complete ground-up PyTorch implementation of the classic **AlexNet** architecture introduced in *ImageNet Classification with Deep Convolutional Neural Networks (2012)*.

Paper: [https://arxiv.org/abs/1404.5997](https://arxiv.org/abs/1404.5997)

---

## Overview

This repository rebuilds AlexNet manually instead of importing torchvision models. The goal is educational clarity — every convolution, activation, pooling operation, and training step is explicitly written to understand how CNNs actually work internally.

The implementation is adapted to modern datasets (like CIFAR‑10) while preserving the architectural philosophy of the original network.

---

## Features

* Manual AlexNet architecture implementation
* Adaptive pooling to support smaller images (32×32 datasets)
* Custom training loop
* Accuracy and loss tracking
* Architecture visualization tools
* Training curve plotting utilities
* GPU compatible training

---

## Architecture Details

The architecture follows the classical AlexNet design adapted for smaller input resolutions using **AdaptiveAvgPool2d(6×6)** to maintain a fixed feature size before the classifier.

### Feature Extractor

| Layer           | Configuration                          | Parameters |
| --------------- | -------------------------------------- | ---------- |
| Conv1           | 3 → 96, kernel=11, stride=4, padding=4 | 34,944     |
| ReLU            | Activation                             | 0          |
| Conv2           | 96 → 256, kernel=5, padding=2          | 614,656    |
| ReLU            | Activation                             | 0          |
| MaxPool         | kernel=3, stride=2                     | 0          |
| Conv3           | 256 → 384, kernel=3, padding=1         | 885,120    |
| ReLU            | Activation                             | 0          |
| Conv4           | 384 → 384, kernel=3, padding=1         | 1,327,488  |
| ReLU            | Activation                             | 0          |
| Conv5           | 384 → 256, kernel=3, padding=1         | 884,992    |
| ReLU            | Activation                             | 0          |
| MaxPool         | kernel=3, stride=2                     | 0          |
| AdaptiveAvgPool | Output size = 6×6                      | 0          |

### Classifier

| Layer   | Configuration  | Parameters |
| ------- | -------------- | ---------- |
| Dropout | Regularization | 0          |
| FC1     | 9216 → 4096    | 37,752,832 |
| ReLU    | Activation     | 0          |
| Dropout | Regularization | 0          |
| FC2     | 4096 → 4096    | 16,781,312 |
| ReLU    | Activation     | 0          |
| FC3     | 4096 → 10      | 40,970     |

### Parameter Summary

Total Parameters: **58,322,314**

Trainable Parameters: **58,322,314**

Non-trainable Parameters: **0**

---

## Dataset

Default: CIFAR‑10

The network automatically adjusts spatial resolution using adaptive pooling, so resizing to 224×224 is not required.

---
### Training Curves

Below figures show the training dynamics over 50 epochs.

These plots demonstrate stable convergence with diminishing loss and saturating accuracy typical for AlexNet‑style CNNs on CIFAR‑10.

Training Loss vs Epoch
<img width="965" height="685" alt="training_loss" src="https://github.com/user-attachments/assets/45a608c4-7eff-4531-9064-33d25ad951d2" />

Training Accuracy vs Epoch
<img width="965" height="685" alt="training_accuracy" src="https://github.com/user-attachments/assets/217eea2e-f81b-428c-ba7e-86a6c32b0992" />



---

## Architecture Visualization

The project can automatically generate a vertical research‑style architecture diagram of the model without requiring dummy inputs.

Output:

<img width="173" height="3473" alt="alexnet_paper" src="https://github.com/user-attachments/assets/206a27e0-7d6a-43c4-9f50-99461e2eac49" />


---

## Why Build From Scratch?

Modern libraries hide most internal mechanics. Reimplementing classic networks provides intuition about:

* receptive fields
* feature hierarchies
* tensor dimensionality flow
* training stability
* computational cost

Understanding these fundamentals helps when designing new architectures or debugging deep models.

---

## Results

Dataset: **CIFAR-10**
Implementation file: **AlexNet_from_Scratch.ipynb** (single notebook)

**Test Accuracy:** 84.03%



### Parameter Count

**Total Parameters:** 58,322,314
**Trainable Parameters:** 58,322,314
**Non‑trainable Parameters:** 0

---

## Repository Contents

```
AlexNet_from_Scratch.ipynb
training_loss.png
training_accuracy.png
```

The entire implementation — model definition, training loop, evaluation, plotting, and visualization — is contained inside a single notebook.

Krizhevsky, A., Sutskever, I., & Hinton, G. (2012)
ImageNet Classification with Deep Convolutional Neural Networks
[https://arxiv.org/abs/1404.5997](https://arxiv.org/abs/1404.5997)

---

## License

MIT — free to use for learning and experimentation.

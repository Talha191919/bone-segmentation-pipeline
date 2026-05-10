# Deep Learning for Bone Segmentation in Medical Images 🦴

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.3.0-green.svg)](https://monai.io/)

A deep learning pipeline for automated bone segmentation in medical imagery, designed with modularity, scalability, and reproducibility in mind. 

## 📌 Overview
This repository implements a robust training and evaluation pipeline for bone structure segmentation. Utilizing **U-Net (ResNet34 backbone)** and **MONAI's DiceCELoss**, the project features custom data handling, z-score normalization, and a rigorous **5-Fold Cross-Validation** strategy to ensure high generalization capability.

## 📁 Repository Structure
```text
.
├── src/
│   ├── dataset.py      # Custom PyTorch Dataset (Z-norm, Augmentation, Matching)
│   └── utils.py        # Core metrics (Dice, Hausdorff) and visualization tools
├── train.py            # Main training loop with 5-Fold CV integration
├── requirements.txt    # Project dependencies
└── README.md
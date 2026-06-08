# VGG16 for Fruit Recognition

**Classify 15 different fruit species using a fine-tuned VGG16 model with a custom weighted categorical cross-entropy loss.**

This repository contains the complete implementation of a deep learning project for the **Fruit Recognition dataset** (Kaggle: chrisfilo/fruit-recognition). The goal is to classify fruit images into 15 distinct classes with high accuracy using transfer learning.

## Key Features
- **Base Model**: VGG16 pre-trained on ImageNet
- **Custom Loss Function**: Weighted Categorical Cross-Entropy to handle class imbalance
- **Data Augmentation**: Rotation, zoom, shift, flip for robust training
- **Evaluation**: Accuracy, loss curves, confusion matrix, per-class metrics
- **Documentation**: Full methodology, training process, and results

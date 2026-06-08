# 🍎 VGG16 for Fruit Recognition

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Classify 15 different fruit species using a fine‑tuned **VGG16** model with **custom loss functions** to tackle **class imbalance** and **inter‑class similarity**.

---

## 👥 Team Members & Contributions

| Name | Roll Number | Contributions |
|------|-------------|---------------|
| **Shravan K.** | CS24B1023 | Data organisation, dataset restructuring, baseline CCE experiment, Weighted Categorical Cross‑entropy experiment |
| **Rohit Man** | AD24B1054 | Focal Loss experiment, comparative analysis, project documentation, GitHub repository management, literature review (Focal Loss paper) |

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Source** | [Fruit Recognition dataset on Kaggle](https://www.kaggle.com/datasets/chrisfilo/fruit-recognition) |
| **Classes** | 15 (Apple, Banana, Carambola, Guava, Kiwi, Mango, Orange, Peach, Pear, Persimmon, Pitaya, Plum, Pomegranate, Tomatoes, muskmelon) |
| **Total images** | 44,406 |
| **Train / Validation / Test** | 70% / 15% / 15% |

---

## 🧠 Methodology

- **Base model**: VGG16 pre‑trained on ImageNet – convolutional layers frozen, custom classification head added.
- **Image size**: 128×128 (fits 4 GB GPU memory).
- **Batch size**: 8.
- **Data augmentation**: rotation, width/height shift, zoom, horizontal flip.
- **Mixed precision**: `mixed_float16` – faster training on RTX 3050.
- **Custom loss functions** (core of the project):
  1. **Weighted Categorical Cross‑entropy** – higher weight for minority classes, forcing the model to learn rare fruits.
  2. **Focal Loss** (`γ=2.0, α=0.25`) – down‑weights easy examples, focuses on hard‑to‑distinguish pairs (paper identified by Rohitman).

---

## 📈 Results

| Loss Function               | Test Accuracy | Minority Recall | Best Val Accuracy |
|-----------------------------|---------------|-----------------|-------------------|
| CCE (baseline)              | 83.71%        | 0.798           | 83.41%            |
| **Weighted CE (custom)**    | 81.75%        | **0.930**       | 81.46%            |
| **Focal Loss (custom)**     | **84.38%**    | 0.807           | 84.24%            |

- **Weighted CE** dramatically improves recall on minority classes (from 0.798 → 0.930), at a small cost in overall accuracy – ideal for imbalanced data.
- **Focal Loss** achieves the highest test accuracy and reduces confusion between similar fruits.

---

## 🖼️ Visualisations

All training curves and confusion matrices are inside the Jupyter notebooks:

- `Exp-1_CCE.ipynb` – baseline
- `Exp-2_WCE.ipynb` – Weighted CE
- `Exp-3_FocalLoss.ipynb` – Focal Loss
- `Exp-4_Comparison.ipynb` – side‑by‑side comparison of all three

---

## 🐳 How to Run (Docker + WSL2)

1. **Clone the repository**  
   ```bash
   git clone https://github.com/Slk2006/ICS-vgg16-fruit-classification.git
   cd ICS-vgg16-fruit-classification
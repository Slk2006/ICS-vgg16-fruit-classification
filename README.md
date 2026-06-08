# 🍎 VGG16 for Fruit Recognition

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Classify **15 different fruit species** using a fine-tuned **VGG16** model with **custom loss functions** designed to tackle **class imbalance** and **inter-class similarity**.

---

## 📋 Table of Contents

- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Results](#-results)
- [Visualisations](#️-visualisations)
- [How to Run](#-how-to-run-docker--wsl2)
- [Repository Structure](#-repository-structure)
- [Conclusion](#-conclusion)
- [References](#-references)

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | [Fruit Recognition — Kaggle (chrisfilo)](https://www.kaggle.com/datasets/chrisfilo/fruit-recognition) |
| **Classes** | 15 — Apple, Banana, Carambola, Guava, Kiwi, Mango, Orange, Peach, Pear, Persimmon, Pitaya, Plum, Pomegranate, Tomato, Muskmelon |
| **Total images** | 44,406 |
| **Split** | 70% Train / 15% Validation / 15% Test |

---

## 🧠 Methodology

### Base Architecture

- **Model**: VGG16 pre-trained on ImageNet
- Convolutional layers **frozen**; custom classification head appended
- **Input size**: 128 × 128 (fits within 4 GB GPU VRAM)
- **Batch size**: 8

### Training Setup

| Setting | Value |
|---|---|
| Mixed precision | `mixed_float16` (optimised for RTX 3050) |
| Optimizer | Adam |
| Epochs | Early stopping with validation monitoring |

### Data Augmentation

Applied to the training generator to improve generalisation:

- Random rotation
- Width & height shift
- Zoom
- Horizontal flip

### Custom Loss Functions

The core contribution of this project — two alternatives to standard categorical cross-entropy (CCE):

**1. Weighted Categorical Cross-Entropy (WCE)**  
Assigns higher loss weight to minority (rare) classes, forcing the model to learn under-represented fruit types rather than defaulting to majority-class predictions.

**2. Focal Loss** (`γ = 2.0, α = 0.25`)  
Down-weights well-classified (easy) examples and focuses training effort on hard, ambiguous samples — particularly useful for visually similar fruit pairs (e.g., Peach vs. Mango).

---

## 📈 Results

| Loss Function | Test Accuracy | Minority Recall | Best Val Accuracy |
|---|---|---|---|
| CCE *(baseline)* | 83.71% | 0.798 | 83.41% |
| **Weighted CE** *(custom)* | 81.75% | **0.930** | 81.46% |
| **Focal Loss** *(custom)* | **84.38%** | 0.807 | 84.24% |

### Key Takeaways

- **Weighted CE** dramatically improves minority-class recall (0.798 → **0.930**) at a modest cost in overall accuracy. Best suited for **severely imbalanced datasets**.
- **Focal Loss** achieves the highest test accuracy and reduces confusion between visually similar fruits. Best suited when **inter-class similarity** is the primary challenge.

---

## 🖼️ Visualisations

Training curves and confusion matrices are available in the notebooks below:

| Notebook | Contents |
|---|---|
| `Exp-1_CCE.ipynb` | Baseline training — standard CCE |
| `Exp-2_WCE.ipynb` | Weighted CE experiment |
| `Exp-3_FocalLoss.ipynb` | Focal Loss experiment |
| `Exp-4_Comparison.ipynb` | Side-by-side comparison of all three runs |

---

## 🐳 How to Run (Docker + WSL2)

### Prerequisites

- Docker Desktop with WSL2 backend enabled
- NVIDIA GPU with drivers installed
- Kaggle API credentials (`kaggle.json`)

### Steps

**1. Clone the repository**

```bash
git clone https://github.com/Slk2006/ICS-vgg16-fruit-classification.git
cd ICS-vgg16-fruit-classification
```

**2. Download the dataset**

Use the Kaggle API or follow the download steps inside `Basic.ipynb`.

**3. Start the Docker container**

Ensure Docker Desktop is running, then:

```bash
docker run -it --rm --gpus all \
  -p 8888:8888 \
  -v "$(pwd):/tf/work" \
  tensorflow/tensorflow:2.15.0-gpu-jupyter
```

**4. Open Jupyter**

Copy the URL printed in the terminal (e.g., `http://127.0.0.1:8888/lab?token=...`) into your browser.

**5. Run the notebooks in order**

| Order | Notebook | Purpose |
|---|---|---|
| 1 | `Basic.ipynb` | Data flattening, splitting, generator setup |
| 2 | `Exp-1_CCE.ipynb` | Baseline experiment |
| 3 | `Exp-2_WCE.ipynb` | Weighted CE experiment |
| 4 | `Exp-3_FocalLoss.ipynb` | Focal Loss experiment |
| 5 | `Exp-4_Comparison.ipynb` | Comparison plots and summary table |

---

## 📁 Repository Structure

```
.
├── Basic.ipynb               # Data flattening, splitting, generators
├── Exp-1_CCE.ipynb           # Baseline experiment
├── Exp-2_WCE.ipynb           # Weighted CE experiment
├── Exp-3_FocalLoss.ipynb     # Focal Loss experiment
├── Exp-4_Comparison.ipynb    # Comparison plots and summary table
├── Dockerfile                # Optional custom image
├── .gitignore
└── README.md
```

---

## 🏁 Conclusion

Both custom loss functions successfully address the limitations of standard categorical cross-entropy:

- **Weighted Categorical Cross-Entropy** is the better choice when the dataset has severe class imbalance and high recall on minority classes is critical.
- **Focal Loss** is advantageous when similar-looking classes cause confusion, delivering the highest overall accuracy.

Together, these experiments demonstrate a complete deep learning pipeline: data preparation, transfer learning with VGG16, custom loss implementation, training, evaluation, and multi-experiment comparison.

---

## 📚 References

1. Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition*. [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)
2. Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). *Focal Loss for Dense Object Detection*. ICCV 2017.
3. Kaggle Fruit Recognition dataset by [chrisfilo](https://www.kaggle.com/datasets/chrisfilo/fruit-recognition).

---
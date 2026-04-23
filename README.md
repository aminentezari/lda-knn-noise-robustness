# Exploring the Interaction Between Noise Perturbation and Regularization in LDA-KNN for Digit Recognition

Bachelor's thesis in Data Analytics — Università degli Studi della Campania "Luigi Vanvitelli"

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)

---

## Overview

This thesis investigates how image noise and regularization interact in a **white-box** Regularized Linear Discriminant Analysis (R-LDA) + K-Nearest Neighbors (KNN) pipeline for handwritten digit recognition on the MNIST dataset.

Classical LDA fails in high-dimensional settings because the within-class scatter matrix becomes ill-conditioned. This work implements a custom **ridge-regularized LDA** from scratch and stress-tests it against five noise types at varying intensities, exploring how the regularization parameter τ can recover classification performance under noisy conditions.

---

## Key Findings

| Noise Type | Accuracy Drop (max) | Sensitivity to τ |
|---|---|---|
| Salt & Pepper | 81.5% | High |
| Gaussian | 74.8% | High |
| Zigzag | 41.1% | Low |
| Blur | 34.3% | Moderate |
| Speckle | 5.5% | Moderate |

- **Baseline accuracy (clean MNIST):** 91.75% with k=7 and τ = 10⁻⁴
- **Best regularization:** τ = 10⁻⁴ consistently outperformed all other values across all noise types
- **Most robust to noise:** Speckle — maintained >86% accuracy even at high intensity
- **Most vulnerable:** Salt & Pepper — pixel-level corruption directly disrupts LDA class separability
- **Surprising result:** Increasing zigzag frequency improved robustness, suggesting periodic distortions introduce learnable patterns

---

## Methodology

**Pipeline:**
```
Clean MNIST (70/30 split) → R-LDA training → Projection matrix Q ∈ ℝ⁷⁸⁴ˣ⁹
→ Noise injected on test set only → Project noisy test via Q → KNN classification (k=7)
```

**Regularization strategy:**

The within-class scatter matrix is stabilized via ridge regularization:

`S_W^(reg) = S_W + τI`, where `τ = ε / d₁²`

**Noise types tested:**
- **Gaussian** — additive zero-mean noise: `X_noisy = clip(X + N(0,σ²), 0, 1)`
- **Salt & Pepper** — binary pixel corruption at density d
- **Gaussian Blur** — convolution with Gaussian kernel (σ = 0.5 to 2.0)
- **Zigzag** — sinusoidal row-wise displacement: `x' = x + A·sin(2πF·y/H)`
- **Speckle** — multiplicative noise: `X_noisy = X + X ⊙ N(0,σ²)`

---

## Dataset

- **MNIST** — 70,000 grayscale 28×28 images (digits 0–9)
- Split: 49,000 training / 21,000 test (stratified 70/30)
- Normalized to [0,1]; noise applied **only to test set**
- Features: 784 (flattened) → reduced to 9 via R-LDA (C−1 components)

---

## Project Structure

```
my-thesis-project/
│
├── README.md
│
├── src/
│   ├── lda_core.py               # Core R-LDA white-box implementation
│   ├── lda_baseline.py           # Baseline R-LDA + KNN on clean MNIST
│   ├── lda_noise_pipeline.py     # Noise injection pipeline
│   ├── lda_final_test.py         # Consolidated experiment runner
│   └── noise/
│       ├── lda_gaussian.py       # Gaussian noise experiments
│       ├── lda_salt_pepper.py    # Salt & pepper experiments
│       ├── lda_speckle.py        # Speckle noise experiments
│       ├── lda_zigzag.py         # Zigzag distortion experiments
│       └── lda_blur.py           # Gaussian blur experiments
│
└── plots/                        # Output visualizations
```

---

## Supervisor

Prof. Rosanna Campagna · Co-supervisor: Prof. Antonio Balzanella  
Università degli Studi della Campania "Luigi Vanvitelli" · A.Y. 2024/2025
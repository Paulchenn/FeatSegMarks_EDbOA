# FeatSegMarks: Edge- & Shape-Deformation-based Online Augmentation (EDbOA / SDbOA)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Academic Project](https://img.shields.io/badge/Masterthesis-HKA%20RKIM-green.svg)](https://www.h-ka.de/)

> **Master's Thesis Project** at **Hochschule Karlsruhe (HKA)** – Institute for Robotics and Autonomous Systems (IN-RAS / RKIM).  
> Reproduction, architectural enhancement, and evaluation of **Shape-Deformation-based Online Augmentation (SDbOA)** / **Edge Deformation-based Online Augmentation (EDbOA)** for robust computer vision and shape-biased representation learning.

---

## Table of Contents

1. [Project Overview & Motivation](#project-overview--motivation)
2. [Methodology & Architecture](#methodology--architecture)
   - [Core Augmentation Pipeline](#core-augmentation-pipeline)
   - [Key Modules (DS-EMSE, TSD, TSG)](#key-modules)
   - [Two-Stage Training Scheme](#two-stage-training-scheme)
3. [Repository Structure](#repository-structure)
4. [Prerequisites & Installation](#prerequisites--installation)
5. [Datasets & Data Preparation](#datasets--data-preparation)
6. [Configuration Reference](#configuration-reference)
7. [Step-by-Step Workflow](#step-by-step-workflow)
   - [1. Interactive Edge Tuning](#1-interactive-edge-tuning)
   - [2. Generator Training (ImageNet / MegaDepth)](#2-generator-training-imagenet--megadepth)
   - [3. Generator Diagnostics & Sensitivity Analysis](#3-generator-diagnostics--sensitivity-analysis)
   - [4. Classifier Training with Online Augmentation](#4-classifier-training-with-online-augmentation)
   - [5. Robustness & Shape-Bias Evaluation](#5-robustness--shape-bias-evaluation)
   - [6. Live Training Monitoring](#6-live-training-monitoring)
8. [Helper Scripts & Utilities](#helper-scripts--utilities)
9. [Model Architecture Zoo](#model-architecture-zoo)
10. [References & Citations](#references--citations)

---

## Project Overview & Motivation

Standard Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) trained on standard image datasets (e.g., ImageNet) often suffer from a severe **texture bias**: they exploit local superficial textures, surface patterns, and color distributions rather than understanding global geometric shapes. Consequently, standard models fail when confronted with:
- Out-of-distribution textures or stylizations (e.g., sketches, silhouettes, watercolor),
- Adversarial perturbations that alter high-frequency pixel statistics without affecting human semantic perception,
- Environmental shifts in real-world robotics and autonomous systems.

Inspired by the ICCV work *"Shift from Texture-bias to Shape-bias: Edge Deformation-based Augmentation for Robust Object Recognition"* (Ge et al.), this project implements an **online data augmentation pipeline** based on **non-linear shape and edge deformations**.

By disassociating shape geometry from texture appearance using a conditional GAN generator, our framework applies non-rigid coordinate deformations to structural edges while re-synthesizing photorealistic textures. Downstream classifiers trained with this online generator are actively guided toward learning invariant, **shape-biased** visual features.

```
+---------------+     +---------------+     +---------------+     +--------------------+
|  Input Image  | --> |    DS-EMSE    | --> |      TSD      | --> |        TSG         |
|      (X)      |     | (Edge Encode) |     |  (TPS Deform) |     | (Generative Recon) |
+---------------+     +---------------+     +---------------+     +--------------------+
        |                                                                    |
        |                      Online Augmentation                           v
        +---------------------------------------------------------> Downstream CNN /
                                                                    ResNet Classifier
```

---

## Methodology & Architecture

### Core Augmentation Pipeline

The overall framework decomposes an input image into high-level structural contours and texture statistics, performs elastic spatial deformation on the shape channel, and re-renders the augmented image:

1. **Edge Extraction (DS-EMSE / EMSE):** Extracts salient structural boundaries while suppressing noisy high-frequency texture details.
2. **Shape Warping (TSD):** Uses Thin-Plate Splines (TPS) to bend, stretch, and compress the extracted contours via randomized control point displacement.
3. **Conditioned Generation (TSG):** Generates photorealistic images that strictly adhere to the deformed contours while retaining the texture identity of the original sample.
4. **Online Training Loop:** Injects deformed samples on-the-fly during classifier optimization, creating virtually infinite diverse shape variations.

### Key Modules

Located in [`utils/modules.py`](file:///Users/glaserpaul/Documents/010_HKA_RKIM/Masterthesis/FeatSegMarks_EDbOA/utils/modules.py):

* **`DS_EMSE` (Differentiable Soft Salient Edge Map Shape Encoding):**  
  A fully differentiable edge-extraction module featuring:
  - Gaussian pre-blurring ($\sigma$, kernel $k$) to eliminate fine texture noise,
  - Sobel filter convolutions for gradient magnitude calculation,
  - Soft sigmoid thresholding ($\tau$, $\beta$) for continuous gradient backpropagation,
  - Morphological opening and closing operations to prune artifacts and reinforce solid object silhouettes.
* **`EMSE` (Edge Map-based Shape Encoding):**  
  Traditional edge extraction based on Canny edge detection with dynamic and Otsu thresholding.
* **`TSD` (TPS-based Shape Deformation):**  
  Parametric non-rigid transformation using Thin-Plate Splines (via [`utils/tps_grid_gen.py`](file:///Users/glaserpaul/Documents/010_HKA_RKIM/Masterthesis/FeatSegMarks_EDbOA/utils/tps_grid_gen.py)). Perturbs control point grids with configurable variance $\lambda$ (uniform or normal distribution).
* **`TSG` (Texture and Shape-based Generation):**  
  Conditional GAN pipeline combining a high-capacity Generator and Discriminator conditioned on edge maps and downsampled/texture-dropped image inputs.

### Two-Stage Training Scheme

The generator and discriminator are trained in two consecutive phases (controlled via `start_in_stage` in [`config/config.json`](file:///Users/glaserpaul/Documents/010_HKA_RKIM/Masterthesis/FeatSegMarks_EDbOA/config/config.json)):

| Stage | Objective | Active Losses | Purpose |
|---|---|---|---|
| **Stage 1** | Reconstruction & Structure Mapping | $\mathcal{L}_{\text{L1}}$, $\mathcal{L}_{\text{edge}}$, $\mathcal{L}_{\text{GAN}}$, $\mathcal{L}_{\text{CE}}$ | Teaches the generator to reconstruct clean images strictly guided by the edge contours and conditioned textures. |
| **Stage 2** | Adversarial Refinement & Semantic Guidance | $\mathcal{L}_{\text{L1}}$, $\mathcal{L}_{\text{edge}}$, $\mathcal{L}_{\text{GAN}}$, $\mathcal{L}_{\text{CE}}$, $\mathcal{L}_{\text{cls}}$ | Fine-tunes realistic synthesis under TPS deformation with classifier guidance ($\mathcal{L}_{\text{cls}}$) ensuring class identity preservation. |

---

## Repository Structure

```text
FeatSegMarks_EDbOA/
├── config/                         # Configuration files (JSON)
│   ├── config.json                 # Main ImageNet training configuration
│   ├── config_cifar10.json         # CIFAR-10 training configuration
│   └── config_megaDepth.json       # MegaDepth high-resolution configuration
├── helper_scripts/                 # Setup, pre-processing, and GUI tools
│   ├── edge_tuner.py               # Interactive Tkinter GUI for DS-EMSE edge tuning
│   ├── build_imagenet_config.py    # Auto-generates ImageNet synset configs
│   ├── convert_cifar.py            # Converts CIFAR batches to structured folders
│   ├── deform_image.py             # Single-image deformation test utility
│   ├── debug_forward.py            # Model forward-pass sanity checker
│   ├── rename_epochs.py            # Renames checkpoint files when chaining runs
│   └── shift_epochs_csv.py         # Adjusts epoch offsets in CSV metrics logs
├── models/                         # Neural network architectures
│   ├── generation_imageNet_V1.py   # Baseline deconv-based generator
│   ├── generation_imageNet_V2_1.py # VGG-encoder-decoder generator
│   ├── generation_imageNet_V2_2.py # Enhanced V2 architecture with reflection padding
│   ├── generation_imageNet_V2_3.py # High-res adapted architecture (MegaDepth)
│   ├── generation_imageNet_V3_1.py # SegNet-style transposed conv architecture
│   ├── generation_cifar10.py       # Lightweight model for CIFAR-10
│   └── resnet.py                   # Custom ResNet-18 classifier implementation
├── utils/                          # Shared utilities and core algorithms
│   ├── modules.py                  # DS_EMSE, EMSE, TSD, TSG classes
│   ├── helpers.py                  # Dataloaders, schedulers, checkpoint handlers
│   ├── tps_grid_gen.py             # Thin-Plate Spline (TPS) transformation engine
│   ├── canny.py & canny_utils.py   # Differentiable Canny operator implementations
│   ├── EdgeExtraction.py           # Traditional filter operations
│   ├── get_ImageNet.py             # ImageNet filtering and synset selector
│   └── preprocess_tinyImageNet.py  # Directory formatter for TinyImageNet
├── train_SDbOA_imageNet.py         # Main training script for ImageNet256
├── train_SDbOA_megaDepth.py        # Main training script for MegaDepth
├── train_cls_with_generator.py     # Downstream classifier training with online generator
├── diagnostics.py                  # Sensitivity analysis & edge-usage diagnostics
├── eval_script.py                  # ICCV-standard evaluation & robustness benchmark
├── live_gan_plot.py                # Real-time Matplotlib loss and metric monitor
├── requirements.txt                # Python package dependencies
└── README.md                       # Project documentation
```

---

## Prerequisites & Installation

### Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Paulchenn/HKA_FEProjekt2025_FeatSegMarks_IccvEDbOA.git FeatSegMarks_EDbOA
   cd FeatSegMarks_EDbOA
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

> [!NOTE]
> Training with large batch sizes or high resolutions (e.g., MegaDepth $608 \times 800$) requires an NVIDIA GPU with at least 12 GB VRAM and CUDA $\ge 11.8$. Mixed-precision training (`torch.amp.autocast`) is integrated by default.

---

## Datasets & Data Preparation

### 1. ImageNet / ImageNet256
The codebase supports ImageNet subsampled to $256 \times 256$.
- Place your ImageNet dataset under `~/torch/data/ImageNet256/` (or update `dataset.path` in [`config/config.json`](file:///Users/glaserpaul/Documents/010_HKA_RKIM/Masterthesis/FeatSegMarks_EDbOA/config/config.json)).
- Directory structure:
  ```text
  ImageNet256/
  ├── train/
  │   ├── n01440764/
  │   └── ...
  └── val/
      ├── n01440764/
      └── ...
  ```
- To customize or auto-generate selected synsets, run:
  ```bash
  python helper_scripts/build_imagenet_config.py
  ```

### 2. MegaDepth
For high-resolution structural scene deformation ($608 \times 800$), configure [`config/config_megaDepth.json`](file:///Users/glaserpaul/Documents/010_HKA_RKIM/Masterthesis/FeatSegMarks_EDbOA/config/config_megaDepth.json) with your local MegaDepth root folder.

### 3. CIFAR-10
To unpack and convert official Python CIFAR-10 batches into image folders:
```bash
python helper_scripts/convert_cifar.py
```

---

## Configuration Reference

The main hyperparameter file is [`config/config.json`](file:///Users/glaserpaul/Documents/010_HKA_RKIM/Masterthesis/FeatSegMarks_EDbOA/config/config.json):

```jsonc
{
  "device": "cuda",
  "batch_size": 16,
  "epochs": 999,
  "start_in_stage": 1,           // 1 = Reconstruction, 2 = Adversarial + Deformation
  "patience_stage1": 15,          // Early stopping / stage transition patience
  "patience_stage2": 20,

  // Loss weights
  "loss_weights": {
    "lambda_stage1_l1": 6.0,
    "lambda_stage1_edge": 1.5,
    "lambda_stage1_GAN": 1.0,
    "lambda_stage1_CE": 0.5,

    "lambda_stage2_l1": 2.0,
    "lambda_stage2_edge": 3.0,
    "lambda_stage2_GAN": 1.0,
    "lambda_stage2_CE": 0.5,
    "lambda_stage2_cls": 0.05
  },

  // DS-EMSE Edge Extraction Hyperparameters
  "DS_PRE_BLUR_K": 3,            // Gaussian blur kernel size (removes noise)
  "DS_PRE_BLUR_SIGMA": 1.0,      // Gaussian blur sigma
  "DS_TAU": 0.25,                // Soft threshold cutoff parameter
  "DS_BETA": 17.0,               // Sigmoid steepness parameter
  "DS_OPEN_KS": 0,               // Morphological opening kernel (0 = disabled)
  "DS_CLOSE_KS": 0,              // Morphological closing kernel (0 = disabled)

  // TSD Deformation
  "tsd_lam": 0.035,              // TPS displacement factor
  "tsd_dist": "uniform",         // "uniform" or "normal"

  // Texture Dropout Settings
  "tex_dropout_s1": true,
  "texD_p_zero": 0.20,
  "texD_p_heavy": 0.50,
  "texD_down_min": 96,
  "texD_down_max": 160
}
```

---

## Step-by-Step Workflow

### 1. Interactive Edge Tuning

Before training the generator, tune the edge extractor parameters visually to guarantee sharp structural boundaries without distracting texture artifacts:

```bash
python helper_scripts/edge_tuner.py
```

* Adjust **$\tau$ (tau)** and **$\beta$ (beta)** sliders in real time.
* Test morphological filtering (Open / Close / Blur).
* Save tuned parameters directly back into `config.json`.

---

### 2. Generator Training (ImageNet / MegaDepth)

Start the two-stage conditional GAN training:

```bash
# For ImageNet256
python train_SDbOA_imageNet.py

# For MegaDepth (608x800)
python train_SDbOA_megaDepth.py
```

Training checkpoints and visual outputs are automatically managed:
* Model weights and optimizer states are saved to `./Result/<DATASET>/`.
* Rolling checkpoints keep the last $K$ epochs to save disk space while preserving best-stage models.
* Epoch visualization grids (`Stage1_Epoch_XXX.png`, `Stage2_Epoch_XXX.png`) show original images, edge maps, deformations, and reconstructions.

---

### 3. Generator Diagnostics & Sensitivity Analysis

To verify whether the trained generator actually relies on the extracted shape/edge map (rather than falling back to texture memorization), run the diagnostic suite:

```bash
python diagnostics.py \
  --config config/config.json \
  --ckpt ./Result/ImageNet256/best_netG/netG__stage1__Epoch_023.pth \
  --split val \
  --batches 16
```

**Interpretation Rule of Thumb:**
* `edge_underuse_score` $\le 0$: Edge information is actively guiding the generation.
* `grad_ratio_edge_E_over_I` $\ge 0.5$: Significant gradient attribution flows through the shape pathway.

---

### 4. Classifier Training with Online Augmentation

Train a downstream visual classifier (e.g., ResNet-18) using the trained generator as an **online data augmentor**:

```bash
python train_cls_with_generator.py
```

During each iteration, input batches are transformed through the `DS-EMSE -> TSD -> Generator` pipeline in real time, exposing the downstream model to non-rigid geometric variations while penalizing texture-based shortcuts.

---

### 5. Robustness & Shape-Bias Evaluation

Benchmark baseline models against SDbOA-trained models across standard ICCV metrics:

```bash
python eval_script.py \
  --ckpt_baseline ./checkpoints/resnet18_baseline.pth \
  --ckpt_sdboa ./checkpoints/resnet18_sdboa.pth \
  --eps 0.03137 \
  --device cuda
```

The script evaluates and outputs:
- **Clean Accuracy:** Standard test set classification performance.
- **Shape-Bias Metrics:**
  - `sbGE` (Shape-bias Generalization Evaluation): Accuracy on cue-conflict images (where shape and texture conflict).
  - `sbIS` (Shape-bias Invariance Score): Feature correlation across image pairs sharing identical shape but differing textures.
- **Adversarial Robustness:** Accuracy under FGSM, PGD, and DeepFool attacks ($\epsilon = 8/255$).
- **Backdoor Attack Resistance:** Clean accuracy vs. Attack Success Rate (ASR).

---

### 6. Live Training Monitoring

Track GAN training metrics (Generator loss, Discriminator loss, L1 loss, Edge loss) in real time:

```bash
python live_gan_plot.py
```

Automatically detects the latest run in `./Result/` and continuously refreshes loss curves and moving averages.

---

## Helper Scripts & Utilities

| Script | Purpose |
|---|---|
| [`edge_tuner.py`](file:///Users/glaserpaul/Documents/010_HKA_RKIM/Masterthesis/FeatSegMarks_EDbOA/helper_scripts/edge_tuner.py) | Interactive Tkinter GUI for finding optimal Sobel and sigmoid thresholding values. |
| [`deform_image.py`](file:///Users/glaserpaul/Documents/010_HKA_RKIM/Masterthesis/FeatSegMarks_EDbOA/helper_scripts/deform_image.py) | Applies a pretrained generator to a single test image with user-specified TPS parameters. |
| [`build_imagenet_config.py`](file:///Users/glaserpaul/Documents/010_HKA_RKIM/Masterthesis/FeatSegMarks_EDbOA/helper_scripts/build_imagenet_config.py) | Downloads the official torchvision ImageNet class index and exports synset dictionaries. |
| [`convert_cifar.py`](file:///Users/glaserpaul/Documents/010_HKA_RKIM/Masterthesis/FeatSegMarks_EDbOA/helper_scripts/convert_cifar.py) | Converts CIFAR-10 binary batches into `class_name/img_xxx.jpg` folder structures. |
| [`rename_epochs.py`](file:///Users/glaserpaul/Documents/010_HKA_RKIM/Masterthesis/FeatSegMarks_EDbOA/helper_scripts/rename_epochs.py) | Renames numbered epoch artifacts (e.g., adding an offset when resuming training). |
| [`shift_epochs_csv.py`](file:///Users/glaserpaul/Documents/010_HKA_RKIM/Masterthesis/FeatSegMarks_EDbOA/helper_scripts/shift_epochs_csv.py) | Offsets epoch counts in CSV metric logs after training restarts. |

---

## Model Architecture Zoo

Located in [`models/`](file:///Users/glaserpaul/Documents/010_HKA_RKIM/Masterthesis/FeatSegMarks_EDbOA/models/):

* **`generation_imageNet_V1.py`**: Baseline convolutional architecture using anti-aliased BlurPool and transpose convolutions.
* **`generation_imageNet_V2_1.py`**: VGG-16 backbone encoder-decoder with latent space concatenation.
* **`generation_imageNet_V2_2.py`** *(Default for ImageNet256)*: Upgraded VGG encoder-decoder utilizing reflection padding to eliminate boundary artifacts, enhanced residual blocks, and dropout support.
* **`generation_imageNet_V2_3.py`** *(Default for MegaDepth)*: Scaled architecture tailored for non-square high-resolution imagery ($608 \times 800$).
* **`generation_imageNet_V3_1.py`**: SegNet-style transposed decoder variant.
* **`generation_cifar10.py`**: Compact generator/discriminator optimized for $32 \times 32$ images.

---

## References & Citations

If you use or build upon this codebase for your research, please cite the original ICCV paper and this project:

```bibtex
@inproceedings{ge2023shift,
  title     = {Shift from Texture-bias to Shape-bias: Edge Deformation-based Augmentation for Robust Object Recognition},
  author    = {Ge, Chun-Mei and others},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2023}
}
```

```bibtex
@mastersthesis{glaser2025edboa,
  author    = {Paul Glaser},
  title     = {Shape-Deformation-based Online Augmentation for Robust Feature Representation},
  school    = {Hochschule Karlsruhe (HKA) - Institute for Robotics and Autonomous Systems (IN-RAS)},
  year      = {2025}
}
```

---

## License

This project is open-source under the MIT License. See [LICENSE](LICENSE) for details.

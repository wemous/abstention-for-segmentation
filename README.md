# Abstention for Noise-Robust Learning in Medical Image Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repository contains the complete codebase for my Master's Thesis, conducted at the **University of Bonn** in collaboration with the **Fraunhofer Institute IAIS**.

The project introduces a novel, universal abstention framework to enhance the noise-resistance of deep learning models, enabling more reliable training on datasets with noisy labels.

<!-- ![Alt text](assets/ground_truth.png "a title")
![Alt text](assets/dice.png "a title")
![Alt text](assets/ads.png "a title") -->
<!-- ![Qualitative Results Teaser](assets/ground_truth.png) -->


<table>
  <tr>
    <td align="center">
      <img src="assets/ground_truth.png" width="250">
    </td>
    <td align="center">
      <img src="assets/dice.png" width="250">
    </td>
    <td align="center">
      <img src="assets/ads.png" width="250">
    </td>
  </tr>
  <tr>
    <td align="center"><strong>(a) Ground Truth</strong></td>
    <td align="center"><strong>(b) Baseline (Dice Loss)</strong></td>
    <td align="center"><strong>(c) Our Method (ADS)</strong></td>
  </tr>
</table>
This visual comparison on a CaDIS sample with 25% label noise demonstrates how my proposed ADS method produces a significantly cleaner segmentation than the baseline Dice Loss.

---

## Thesis Highlights & Key Achievements

*   **Developed a Universal Abstention Framework:** I designed a modular framework that can be integrated with any loss function to improve its robustness against label noise. The core loss implementations can be found in `losses.py`.
*   **Designed and Implemented Three Novel Loss Functions:** I created the Generalized Abstaining Classifier (GAC), Symmetric Abstaining Classifier (SAC), and Abstaining Dice Segmenter (ADS).
*   **Engineered a Specialized Class-wise Architecture:** For the ADS model, I developed a novel architectural adaptation (in `models/base.py`) to enable class-wise abstention, making the mechanism compatible with region-based losses like Dice Loss.
*   **Built a Reproducible Experimental Pipeline:** I engineered a robust experimental workflow using YAML configurations and a master sweep script (`sweeps/sweep.py`) to systematically train and evaluate models across multiple datasets, noise levels, and random seeds.
*   **Publication-Quality Research:** A scientific paper based on this thesis has been submitted for publication at the IEEE BigData 2025 conference.

---

## Installation

This project was developed using Python 3.12 and PyTorch 2.6 on CUDA 11.8.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/wemous/abstention-for-segmentation.git
    cd abstention-for-segmentation
    ```

2.  **Create a Conda environment (recommended):**
    ```bash
    conda create -n abstention python=3.12
    conda activate abstention
    ```

3.  **Install dependencies:**
    All required packages are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
    ```

---

## Dataset Setup

1.  **Download the datasets:**
    *   **CaDIS:** Available upon request from the author.
    *   **DSAD:** Available on [Figshare](https://springernature.figshare.com/articles/dataset/The_Dresden_Surgical_Anatomy_Dataset_for_abdominal_organ_segmentation_in_surgical_data_science/21702600?file=38494425).

2.  **Organize the data:**
    Create a `data/` directory in the project root and structure the datasets as described in the `datasets` module.

---

## Project Workflow

The codebase is structured to support both quick tests and large-scale, reproducible experiments.

### 1. Quick Sanity Check

For debugging or testing the pipeline with a single model, you can use `trainer.py`. This script is designed for rapid, one-off runs and does not use a config file.

```bash
python trainer.py
```

### 2. Hyperparameter Optimization

I performed hyperparameter tuning for the novel loss functions using Weights & Biases sweeps. The configurations for these sweeps are defined in the YAML files located in the `sweeps/` directory (e.g., `gac_sweep.yaml`).

### 3. Reproducing the Thesis Results

The final, comprehensive results were generated using the master sweep script, which systematically iterates through all experimental conditions.

1.  **Configure the sweep:** Open `config/sweep_config.yaml` to select the dataset(s), losses, and random seeds.
2.  **Run the script:**
    ```bash
    wandb sweep configs/sweep_config.yaml
    wandb agent <sweep-ID>
    ```
    This script trains each model on all noise levels for 5 different random seeds, logging all results to W&B. This is the script that generated the data for the main results table in my thesis.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

This Master's Thesis was completed in collaboration with the **Fraunhofer Institute for Intelligent Analysis and Information Systems (IAIS)**. This research has been funded by the **Federal Ministry of Education and Research of Germany** and the state of **North-Rhine Westphalia** as part of the **Lamarr-Institute for Machine Learning and Artificial Intelligence**.

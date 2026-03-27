<div align="center">

# Generalizing Abstention for Noise-Robust Learning in Medical Image Segmentation

[![MIDL 2026](https://img.shields.io/badge/MIDL_2026-paper-blue)](https://openreview.net/forum?id=0ss826X42Q)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![CI](https://github.com/wemous/abstention-for-segmentation/actions/workflows/linter.yml/badge.svg)](https://github.com/wemous/abstention-for-segmentation/actions/workflows/linter.yml)

</div>

---

This repository contains the official implementation of our **universal abstention framework**, which augments any loss function with the ability to selectively ignore samples the model identifies as likely corrupted — without modifying the underlying architecture.

We introduce two enhancements over prior abstention work ([DAC](https://arxiv.org/abs/1905.10964), [IDAC](https://arxiv.org/abs/2410.21014)):

- **Informed regularization** — guides the abstention rate toward a prior noise estimate η̃, preventing premature collapse to zero abstention.
- **Power-law α auto-tuning** — replaces DAC's stateful linear schedule with a direct, flexible formula controlled by a single growth factor γ.

Using this framework, we derive three novel loss functions: **GAC** (GCE base), **SAC** (SCE base), and **ADS** (Dice base).

---

## Installation

```bash
git clone https://github.com/wemous/abstention-for-segmentation.git
cd abstention-for-segmentation

conda create -n abstention python=3.12 && conda activate abstention

pip install -r requirements.txt
pip install -e .
```

## Datasets

- [**CaDIS**](https://cataracts.grand-challenge.org/CaDIS/) — available upon request from the authors.
- [**DSAD**](https://www.nature.com/articles/s41597-022-01719-2) — available on [Figshare](https://springernature.figshare.com/articles/dataset/The_Dresden_Surgical_Anatomy_Dataset_for_abdominal_organ_segmentation_in_surgical_data_science/21702600).

---

## Results
Average test mIoU (%) across 5 seeds. Noise rate η is the fraction of corrupted pixels.

### CaDIS
 
 
| Loss | 0% | 5% | 10% | 15% | 20% | 25% |
|------|-----|-----|------|------|------|------|
| CE   | 76.02 | 73.67 | 66.39 | 64.15 | 59.56 | 52.27 |
| DAC  | 75.29 | 73.14 | 67.43 | 65.85 | 63.42 | 60.63 |
| IDAC | 75.36 | 72.89 | 66.92 | 64.87 | 60.54 | 58.19 | 
| GCE  | 73.49 | 72.83 | 64.82 | 64.81 | 60.73 | 55.71 |
| GAC ⭐ | 73.76 | 71.73 | 64.16 | 64.44 | 60.91 | 59.46 |
| SCE  | 75.38 | 73.41 | 65.92 | 62.16 | 57.62 | 55.08 |
| SAC ⭐ | 75.83 | 73.51 | 67.29 | 65.48 | 62.70 | 61.27 |
| Dice | 76.52 | 73.48 | 66.51 | 67.31 | 63.64 | 61.04 |
| ADS ⭐ | **77.04** | **75.22** | **71.12** | **70.80** | **68.88** | **66.39** |
 
### DSAD.
 
| Loss | 0% | 3% | 6% | 9% | 12% | 15% |
|------|-----|-----|-----|-----|------|------|
| CE   | 34.25 | 33.69 | 30.70 | 24.65 | 21.00 | 14.41 |
| DAC  | 34.01 | 33.67 | 29.47 | 24.58 | 22.59 | 17.69 |
| IDAC | 33.60 | 32.76 | 29.11 | 23.47 | 20.94 | 16.24 |
| GCE  | **35.14** | **33.84** | 29.69 | 22.95 | 19.84 | 14.12 |
| GAC ⭐ | 32.26 | 32.94 | 29.78 | **28.84** | **25.00** | **20.01** |
| SCE  | 32.78 | 32.11 | 30.51 | 28.02 | 21.57 | 15.31 |
| SAC ⭐ | 33.86 | 30.90 | **31.55** | 28.55 | 23.73 | 15.91 |
| Dice | 31.28 | 30.83 | 28.56 | 19.04 | 16.15 | 14.65 |
| ADS ⭐ | 30.09 | 28.64 | 30.48 | 26.23 | 22.63 | 18.05 |
 
⭐ = our proposed loss functions.

---

## Citation

If you use this code or build on our work, please cite:

```bibtex
@inproceedings{moustafa2026generalizing,
  title     = {Generalizing Abstention for Noise-Robust Learning in Medical Image Segmentation},
  author    = {Moustafa, Wesam and Elsafty, Hossam and Schneider, Helen and Sparrenberg, Lorenz and Sifa, Rafet},
  booktitle = {Medical Imaging with Deep Learning},
  year      = {2026},
  url       = {https://openreview.net/forum?id=0ss826X42Q}
}
```

---

## Acknowledgements

This research was funded by the Federal Ministry of Education and Research of Germany and the state of North-Rhine Westphalia as part of the Lamarr Institute for Machine Learning and Artificial Intelligence.

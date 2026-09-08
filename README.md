# EPC-3D-Diff

[![arXiv](https://img.shields.io/badge/arXiv-2605.20470-b31b1b.svg)](https://arxiv.org/abs/2605.20470)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-blue.svg)](https://www.mozilla.org/MPL/2.0/)
[![MICCAI 2026](https://img.shields.io/badge/MICCAI-2026-4c7bd9.svg)](https://miart-workshop.github.io/)

Official research-code release for **EPC-3D-Diff: Equivariant Physics Consistent Conditional 3D Latent Diffusion for CBCT to CT Synthesis**.

**EPC-3D-Diff** is a conditional 3D latent diffusion framework for volumetric CBCT-to-CT synthesis. The method introduces a **projection-domain rotational equivariance constraint** derived from CT acquisition physics, while retaining efficient **operator-free inference** from CBCT input.

> **Publication status:** accepted to the **Medical Image AI in Radiation Therapy (MIART) Workshop at MICCAI 2026**.  
> **Preprint:** [arXiv:2605.20470](https://arxiv.org/abs/2605.20470)

---

## Overview

Cone-beam CT (CBCT) is widely used for image-guided radiotherapy, but scatter, noise, beam-hardening effects, and reconstruction artefacts can degrade Hounsfield Unit (HU) accuracy. EPC-3D-Diff learns to synthesize CT-quality volumes from CBCT while incorporating acquisition physics as a structural constraint during training.

The framework combines:

- **3D latent conditional diffusion** for volumetric CBCT-to-CT synthesis.
- A **lightweight 3D autoencoder** that preserves axial depth while reducing in-plane spatial resolution.
- A **conditional 3D U-Net** operating on CT and CBCT latent representations.
- **Projection-domain rotational equivariance**, exploiting the relationship between in-plane volume rotation and angular shifts in CT projections.
- **Image-domain structural regularization** using reconstruction, edge, and Laplacian terms.
- **DDIM inference** using only the CBCT input; projection operators are not required at test time.

---

## Method at a glance

Let \(x_0\) denote the reference CT volume and \(x_c\) the paired CBCT volume. A lightweight 3D encoder maps both volumes into compact latent representations \(z_0\) and \(z_c\). Diffusion is performed in latent space, and a conditional 3D U-Net predicts the injected noise using the noisy CT latent, timestep embedding, and CBCT latent condition.

The key physics-consistent component is the projection equivariance constraint. For an in-plane rotation \(R_\phi\), the CT acquisition operator \(A_0\) satisfies the corresponding angular-shift relation in projection space. EPC-3D-Diff therefore encourages the projection of a rotated synthesized CT to agree with the appropriately shifted projection of the reference CT.

After latent denoising, the decoder maps the recovered latent representation back to a synthetic CT volume. During inference, only CBCT is required.

### Methodology diagram

The following diagram provides an overview of the EPC-3D-Diff training and inference workflow:

![EPC-3D-Diff methodology overview](assets/figures/fig1_methodology.png)

*Figure 1. Overview of EPC-3D-Diff. The framework encodes CT and CBCT volumes into a latent space, performs conditional diffusion with CBCT guidance, decodes the recovered latent representation into synthetic CT, and enforces projection-domain rotational equivariance through the physics-based operator pathway during training.*

---

## Repository structure

```text
EPC-3D-Diff/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── train.py
├── test.py
├── diffusion_condition_3d.py
├── datasets.py
├── src/
│   └── misc.py
└── ops/
    ├── astracd_op.py
    ├── astra_autograd.py
    ├── Elekta_ASTRA_Geom.py
    └── Elekta_geometry.xml
```

The current release candidate contains the four uploaded core scripts. The `src/` and `ops/` physics helper modules shown above are required by `train.py` and should be added before the repository is made public.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ALZAHRAALTALIB/EPC-3D-Diff.git
cd EPC-3D-Diff
```

### 2. Create a Python environment

```bash
conda create -n epc3ddiff python=3.10
conda activate epc3ddiff
```

### 3. Install ASTRA Toolbox

EPC-3D-Diff uses ASTRA-based forward projection during physics-consistent training. Install an ASTRA build compatible with your CUDA environment by following the official ASTRA Toolbox installation instructions.

### 4. Install Python dependencies

Install PyTorch for the CUDA version available on your system, then install the remaining packages:

```bash
pip install -r requirements.txt
```

---

## Data

The datasets used in the study are **not distributed in this repository**.

### NWH phantom cohort

The Ninewells (Dundee) phantom cohort contains paired CBCT/CT image-domain data together with raw projection-domain information. The study uses patient-wise training and testing splits.

### JUST clinical cohort

The JUST clinical cohort contains paired head-and-neck CBCT/CT volumes acquired at King Abdullah University Hospital, Jordan. The study also uses patient-wise training and testing splits.

### Preprocessing used in the study

The released preprocessing code includes:

- paired CBCT/CT stack alignment;
- correction for possible stack reversal and in-plane offsets;
- conversion to HU where required;
- clipping to **[-1000, 2000] HU**;
- linear normalization to **[-1, 1]**;
- paired foreground cropping and resizing to **256 × 256**;
- 3D training patch extraction.

Update the dataset paths in `train.py` and `test.py` to match your local data layout.

---

## Training

The paper uses a lightweight latent autoencoder followed by conditional 3D diffusion training.

Before training the diffusion model, ensure that:

1. the required `src/` and `ops/` modules are present;
2. the NWH/JUST dataset paths are configured;
3. the ASTRA geometry files match the acquisition setup;
4. the pretrained latent autoencoder checkpoint is available, or autoencoder pretraining is enabled.

Run:

```bash
python train.py
```

The current script contains configuration variables for:

- dataset selection (`NWH`, `JUST`, or mixed-domain training);
- latent dimensionality;
- diffusion epochs and checkpoint resume;
- physics-consistency terms;
- projection-domain equivariance frequency;
- GPU selection.

For a final public release, these options should be moved to command-line arguments or a YAML configuration file.

---

## Inference and evaluation

Run:

```bash
python test.py
```

The testing pipeline performs 3D latent DDIM sampling and reports quantitative comparisons between synthesized CT and reference CT, including:

- SSIM;
- PSNR;
- MSE;
- MAE;
- HU histograms;
- HU line profiles;
- absolute difference maps.

The test-time synthesis path uses only the CBCT condition. Reference CT is used only for retrospective evaluation.

---

## Experimental setting reported in the paper

The paper reports:

| Setting | Value |
|---|---:|
| Latent channels | 4 |
| Conditional 3D U-Net base width | 64 |
| Timestep embedding | 256 |
| Diffusion timesteps | 1000 |
| Noise schedule | linear, \(10^{-4}\) to \(5\times10^{-3}\) |
| Batch size | 2 |
| Optimizer | Adam |
| Learning rate | \(10^{-5}\) |
| Diffusion training | 2500 epochs |
| DDIM inference | 100 steps |
| Input in-plane size | 256 × 256 |

For mixed-domain training, balanced mini-batches are used to prevent the larger cohort from dominating optimization.

---

## Reported results

On the NWH phantom test set, the paper reports the following multiple-domain training results:

| Method | PSNR (dB) | SSIM |
|---|---:|---:|
| CycleGAN | 30.50 ± 7.27 | 0.88 ± 0.07 |
| C-DDPM | 31.00 ± 4.48 | 0.93 ± 0.05 |
| **EPC-3D-Diff** | **38.44 ± 2.28** | **0.99 ± 0.01** |

The study also reports an average improvement of approximately **+1.8 dB PSNR** over C-DDPM on the JUST clinical test cohort.

Please refer to the paper for the complete quantitative results, ablations, qualitative comparisons, and HU analysis.

---

## Citation

If you use EPC-3D-Diff in your research, please cite:

```bibtex
@article{altalib2026epc3ddiff,
  title   = {EPC-3D-Diff: Equivariant Physics Consistent Conditional 3D Latent Diffusion for CBCT to CT Synthesis},
  author  = {Altalib, Alzahra and Li, Chunhui and Alewaidat, Haytham Ahmad and Alawneh, Khaled Z. and Qandeel, Ahmad Awad and Perelli, Alessandro},
  journal = {arXiv preprint arXiv:2605.20470},
  year    = {2026}
}
```

The citation will be updated with the MICCAI 2026 MIART Springer proceedings information when the final bibliographic record becomes available.

---

## License

This source code is released under the **Mozilla Public License 2.0 (MPL-2.0)**. See [`LICENSE`](LICENSE).

---

## Disclaimer

This repository contains research software. It is **not a medical device**, has not been validated for clinical decision-making, and should not be used directly for patient diagnosis or treatment without appropriate independent validation and regulatory review.

---

## Contact

**Alzahra Altalib**  
School of Science and Engineering, University of Dundee, UK  
Email: 2600129@dundee.ac.uk

For questions regarding the method, implementation, or reproducibility, please open a GitHub issue or contact the corresponding authors.

# Copyright 2026 Alzahra Altalib, University of Dundee
# (Alzahra Altalib) email: 2600129@dundee.ac.uk
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""
Inference and quantitative evaluation utilities for EPC-3D-Diff.

This script performs 3D latent-space DDIM sampling conditioned on CBCT volumes,
decodes synthetic CT volumes to image space, and evaluates image-quality and HU
agreement metrics against paired reference CT data. [AA2026]

Reference
---------
[AA2026] A. Altalib, C. Li, H. A. Alewaidat, K. Z. Alawneh,
A. A. Qandeel, and A. Perelli,
"EPC-3D-Diff: Equivariant Physics Consistent Conditional 3D Latent Diffusion
for CBCT to CT Synthesis," arXiv:2605.20470, 2026.
Accepted to the MIART Workshop at MICCAI 2026.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

from diffusion_condition_3d import ConditionalUNet as ConditionalUNet3D
from datasets import (BalancedBatchSampler, MedicalImageDataset_JUST_DICOM,
                                     MedicalImageDataset_NWH_DICOM, image_denormalization)
from collections import OrderedDict


# ============================================================
# AE (Latent) MUST match the TRAIN script
# Train file: Train_AA22_with_RawDc_fixed _final
# - LATENT_C = 4
# - Encoder downsamples only H,W (keeps D)
# - Decoder upsamples only H,W (keeps D)
# ============================================================

class Latent3DEncoder(nn.Module):
    """
    Lightweight 3D encoder:
      - input  : [B, 1, D, H, W]
      - output : [B, LATENT_C, D, H/2, W/2]
      - downsamples only H and W (keeps depth D)
    """

    def __init__(self, in_channels: int = 1, hidden_channels: int = 32, latent_channels: int = 4):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(hidden_channels, affine=True),
            nn.SiLU(inplace=True),
        )

        # Downsample only H,W (keep D)
        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Sequential(
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(hidden_channels, affine=True),
            nn.SiLU(inplace=True),
        )

        # Output latent channels (LATENT_C)
        self.conv3 = nn.Conv3d(hidden_channels, latent_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        z = self.conv3(x)
        return z


class Latent3DDecoder(nn.Module):
    def __init__(self, in_channels: int,
                 hidden_channels: int = 32,
                 out_channels: int = 1):
        super().__init__()

        # learnable upsampling in H and W
        self.up = nn.ConvTranspose3d(
            in_channels,
            hidden_channels,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            padding=(0, 0, 0)
        )

        # first residual refinement block
        self.resblock1 = nn.Sequential(
            nn.InstanceNorm3d(hidden_channels, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1),
            nn.InstanceNorm3d(hidden_channels, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1),
        )

        # second residual refinement block
        self.resblock2 = nn.Sequential(
            nn.InstanceNorm3d(hidden_channels, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1),
            nn.InstanceNorm3d(hidden_channels, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1),
        )

        # final projection to one channel
        self.out_conv = nn.Conv3d(hidden_channels, out_channels, 3, padding=1)

    def forward(self, z: torch.Tensor, out_size=None) -> torch.Tensor:
        # upsample latent [B,C,D,H/2,W/2] → [B,hidden,D,H,W]
        x = self.up(z)

        # first residual block: x ← x + resblock1(x)
        res1 = self.resblock1(x)
        x = x + res1

        # second residual block: x ← x + resblock2(x)
        res2 = self.resblock2(x)
        x = x + res2

        # final 3×3×3 conv to get 1‑channel output
        x = self.out_conv(x)

        # centre‑crop if out_size is specified
        if out_size is not None:
            D, H, W = out_size
            _, _, D0, H0, W0 = x.shape
            ds, hs, ws = (D0 - D) // 2, (H0 - H) // 2, (W0 - W) // 2
            x = x[:, :, ds:ds + D, hs:hs + H, ws:ws + W]

        return x


# ----- global latent helpers -----
latent_encoder = None
latent_decoder = None


def get_module(m):
    return m.module if isinstance(m, nn.DataParallel) else m


def encode_latent(x):
    assert latent_encoder is not None
    enc = get_module(latent_encoder)
    return enc(x)


def decode_latent(z, out_size):
    assert latent_decoder is not None
    dec = get_module(latent_decoder)
    return dec(z, out_size=out_size)


def load_ae(path, device):
    """
    Matches training save_ae() which stored:
      {"latent_encoder": ..., "latent_decoder": ...}
    """
    ckpt = torch.load(path, map_location=device)
    get_module(latent_encoder).load_state_dict(ckpt["latent_encoder"])
    get_module(latent_decoder).load_state_dict(ckpt["latent_decoder"])


# ============================================================
# Noise scheduler + DDIM sampler (latent space) (same as train)
# ============================================================

class NoiseScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.005):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        ac = self.alphas_cumprod.to(t.device)[t]  # (B,)

        n_dims = x_start.dim()
        shape = (-1,) + (1,) * (n_dims - 1)

        sqrt_ac = ac.sqrt().view(shape)
        sqrt_om = (1.0 - ac).sqrt().view(shape)
        return sqrt_ac * x_start + sqrt_om * noise


@torch.no_grad()
def sample_ddim(model, condition, noise_scheduler, num_steps=100):
    """
    3D-safe DDIM sampler in LATENT space.

    condition: [B, 1, D, H, W] (image space)
    returns:   [B, 1, D, H, W] (image space)
    """
    model.eval()
    device = condition.device
    B, _, D, H, W = condition.shape

    # encode CBCT to latent
    cond_lat = encode_latent(condition)  # [B,C,D,H/2,W/2]
    x = torch.randn_like(cond_lat, device=device)  # latent noise

    T = noise_scheduler.timesteps
    step = max(T // num_steps, 1)

    t_seq = list(range(0, T, step))
    if t_seq[-1] != T - 1:
        t_seq.append(T - 1)
    t_seq = sorted(t_seq)

    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)

    n_dims = x.dim()
    shape = (1,) * n_dims

    for i in reversed(range(len(t_seq))):
        t = t_seq[i]
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)

        alpha_bar_t = alphas_cumprod[t]
        alpha_bar_t_5 = alpha_bar_t.view(shape)

        eps = model(x, t_batch, cond_lat)

        x0_lat = (x - torch.sqrt(1.0 - alpha_bar_t_5) * eps) / torch.sqrt(alpha_bar_t_5)

        if i == 0:
            x = x0_lat
        else:
            t_prev = t_seq[i - 1]
            alpha_bar_prev = alphas_cumprod[t_prev]
            alpha_bar_prev_5 = alpha_bar_prev.view(shape)

            sigma = torch.sqrt(
                (1.0 - alpha_bar_prev_5) / (1.0 - alpha_bar_t_5)
                * (1.0 - alpha_bar_t_5 / alpha_bar_prev_5)
            )
            z = torch.randn_like(x)
            x = torch.sqrt(alpha_bar_prev_5) * x0_lat + sigma * z

    # decode from latent to image
    x_img = decode_latent(x, out_size=(D, H, W))
    return x_img


# ============================================================
# HU histogram + line profile helpers
# ============================================================

def _finite(vals: np.ndarray) -> np.ndarray:
    vals = np.asarray(vals).ravel()
    return vals[np.isfinite(vals)]


def plot_hu_hist(ct2d, sct2d, save_path, mask=None, bins=400, hu_range=(-1000, 2000), title="HU Histogram"):
    ct2d = np.asarray(ct2d)
    sct2d = np.asarray(sct2d)
    if mask is not None:
        mask = mask.astype(bool)
        ct_vals = _finite(ct2d[mask])
        sct_vals = _finite(sct2d[mask])
    else:
        ct_vals = _finite(ct2d)
        sct_vals = _finite(sct2d)

    plt.figure(figsize=(7, 5))
    plt.hist(ct_vals, bins=bins, range=hu_range, density=True, alpha=0.5, label="CT")
    plt.hist(sct_vals, bins=bins, range=hu_range, density=True, alpha=0.5, label="Synth CT")
    plt.xlabel("HU")
    plt.ylabel("Probability density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    return ct_vals, sct_vals


def line_profile(img2d, p0, p1, n=300):
    """
    Nearest-neighbour sampling along a line (simple + robust).
    p0/p1: (row, col) = (y, x)
    """
    img2d = np.asarray(img2d)
    y0, x0 = p0
    y1, x1 = p1

    ys = np.linspace(y0, y1, n)
    xs = np.linspace(x0, x1, n)

    yi = np.clip(np.round(ys).astype(int), 0, img2d.shape[0] - 1)
    xi = np.clip(np.round(xs).astype(int), 0, img2d.shape[1] - 1)

    vals = img2d[yi, xi]
    dist = np.linspace(0, np.hypot(y1 - y0, x1 - x0), n)
    return dist, vals


def plot_line_profiles(ct2d, sct2d, save_path, p0, p1, n=400, title="HU Line Profile"):
    d, v_ct = line_profile(ct2d, p0, p1, n=n)
    _, v_sct = line_profile(sct2d, p0, p1, n=n)

    plt.figure(figsize=(8, 5))
    plt.plot(d, v_ct, label="CT")
    plt.plot(d, v_sct, label="Synth CT")
    plt.xlabel("Distance (pixels)")
    plt.ylabel("HU")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    # also save absolute error profile
    err_path = os.path.splitext(save_path)[0] + "_abs_err.png"
    plt.figure(figsize=(8, 4))
    plt.plot(d, np.abs(v_sct - v_ct))
    plt.xlabel("Distance (pixels)")
    plt.ylabel("|HU error|")
    plt.title("Absolute HU Error Along Line")
    plt.tight_layout()
    plt.savefig(err_path, dpi=200)
    plt.close()


# ============================================================
# Main test
# (plus extra histogram + line-profile images)
# ============================================================

if __name__ == "__main__":
    # ----- config -----
    DATASET_SELECTED = "NWH"    # Ninewells

    if DATASET_SELECTED == "JUST":
        data_path = "../../Dataset_JUST_np"
    elif DATASET_SELECTED == "NWH":
        data_path = "../../Dataset_Ninewells"

    best_epoch     = 2500  # <<< set the epoch to test
    ckpt_dir       = "checkpoints_{}".format(DATASET_SELECTED)
    checkpoint_path = os.path.join(ckpt_dir, f"model_epoch_{best_epoch}.pth")
    n_steps = 100

    # Bone
    vmin, vmax = -1000, 2000

    # AE checkpoint saved by training script:
    #   checkpoints/ae_latentC4.pth   (LATENT_C=4)
    LATENT_C = 4
    ae_ckpt_path = os.path.join(ckpt_dir, f"ae_latentC{LATENT_C}.pth")

    result_dir = os.path.join(
        os.path.dirname(__file__),
        f"results_3D_EqDiff_{DATASET_SELECTED}_nsteps_{n_steps}_epoch_{best_epoch}"
    )
    os.makedirs(result_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Saving test results to:", result_dir)
    print("Loading model ckpt:", checkpoint_path)
    print("Loading AE ckpt:", ae_ckpt_path)

    # ====== DATASET / LOADER ======
    if DATASET_SELECTED == "JUST":
        dataset = MedicalImageDataset_JUST_DICOM(
            root_dir=data_path,
            mode="test",
            use_3d=True,
            patch_size=(32, 256, 256),
            patches_per_volume=4,
            random_flip_3d=False,
        )
    elif DATASET_SELECTED == "NWH":

        dataset = MedicalImageDataset_NWH_DICOM(
            root_dir=data_path,
            mode="test",
            use_3d=True,
            patch_size=(32, 256, 256),
            patches_per_volume=4,
            random_flip_3d=False,
        )

    test_loader = DataLoader(
        dataset,
        batch_size  = 1,
        shuffle     = True,
        num_workers = 0,
    )

    print("len(dataset):", len(test_loader.dataset))

    # ----- model + AE (match training) -----
    model = ConditionalUNet3D(
        in_channels=LATENT_C,
        cond_channels=LATENT_C,
        base_channels=64,
        time_emb_dim=256,
    ).to(device)

    latent_encoder = Latent3DEncoder(in_channels=1, hidden_channels=32, latent_channels=LATENT_C).to(device)
    latent_decoder = Latent3DDecoder(in_channels=LATENT_C, hidden_channels=32, out_channels=1).to(device)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs for testing")
        model = nn.DataParallel(model)

    # ----- load model checkpoint (training saves dict with "model" or raw state_dict) -----
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        print("Loaded NEW style checkpoint (model dict in 'model').")
        model_state = ckpt["model"]
    else:
        print("Loaded OLD style checkpoint (raw model.state_dict()).")
        model_state = ckpt

    first_key = next(iter(model_state.keys()))
    if isinstance(model, nn.DataParallel) and not first_key.startswith("module."):
        model_state = OrderedDict((f"module.{k}", v) for k, v in model_state.items())
    elif not isinstance(model, nn.DataParallel) and first_key.startswith("module."):
        model_state = OrderedDict((k.replace("module.", ""), v) for k, v in model_state.items())

    model.load_state_dict(model_state)
    print("Loaded model weights.")

    # ----- load AE checkpoint (separate file) -----
    load_ae(ae_ckpt_path, device)
    get_module(latent_encoder).eval()
    get_module(latent_decoder).eval()
    for p in get_module(latent_encoder).parameters():
        p.requires_grad = False
    for p in get_module(latent_decoder).parameters():
        p.requires_grad = False

    # ----- scheduler -----
    scheduler = NoiseScheduler(timesteps=1000)
    HU_MIN, HU_MAX = -1000, 2000

    metrics_log = []
    global_slice_idx = 0

    with torch.no_grad():
        for vol_idx, (cbct_vol, ct_vol, patient_id, domain_id) in enumerate(
                tqdm(test_loader, desc="Processing 3D volumes")
        ):
            cbct_vol = cbct_vol.to(device)  # [1,1,D,H,W]
            ct_vol = ct_vol.to(device)  # [1,1,D,H,W]

            print("Volume index:", vol_idx)
            print("Volume shape:", cbct_vol.shape)

            synth_vol = sample_ddim(model, cbct_vol, scheduler, num_steps=n_steps)  # [1,1,D,H,W]

            cbct_np = image_denormalization(cbct_vol[0].cpu().squeeze().numpy())
            ct_np = image_denormalization(ct_vol[0].cpu().squeeze().numpy())
            synth_np = image_denormalization(synth_vol[0].cpu().squeeze().numpy())

            D = ct_np.shape[0]
            mid = D // 2

            # --- save mid-slice visual per volume  ---
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for ax, img, title in zip(
                    axes,
                    [cbct_np[mid], ct_np[mid], synth_np[mid]],
                    ["CBCT Input", "Ground Truth CT", "Synthesized CT"],
            ):
                ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
                ax.set_title(title)
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, f"test_volume_{vol_idx}_mid.png"))
            plt.close(fig)

            # --- NEW: HU histogram + line profile on mid slice ---
            # Centre crop 256×256 like your metrics ROI
            Hm, Wm = ct_np[mid].shape
            crop_h, crop_w = 256, 256
            h0m = (Hm - crop_h) // 2
            w0m = (Wm - crop_w) // 2

            ct_mid_roi = ct_np[mid][h0m:h0m + crop_h, w0m:w0m + crop_w]
            synth_mid_roi = synth_np[mid][h0m:h0m + crop_h, w0m:w0m + crop_w]

            ct_val, sct_val = plot_hu_hist(
                ct_mid_roi, synth_mid_roi,
                save_path=os.path.join(result_dir, f"hu_hist_vol_{vol_idx}_midROI.png"),
                bins=400,
                hu_range=(HU_MIN, HU_MAX),
                title=f"HU Histogram (mid-slice ROI) | vol={vol_idx} | {patient_id[0] if isinstance(patient_id, (list, tuple)) else patient_id}"
            )

            # a, b are 1D numpy arrays of the same length
            data_HU = np.column_stack((ct_val, sct_val))  # shape: (N, 2)
            out_path = os.path.join(result_dir, f"hu_values_{vol_idx}_midROI.txt")
            np.savetxt(
                out_path,
                data_HU,
                fmt="%.6f",  # or "%d" for ints
                delimiter="\t",  # "\t" for tab, "," for csv-style
                header="col1\tcol2",
                comments=""
            )

            # Default line: horizontal through ROI center
            cy, cx = crop_h // 2, crop_w // 2
            p0 = (cy, 0)
            p1 = (cy, crop_w - 1)
            plot_line_profiles(
                ct_mid_roi, synth_mid_roi,
                save_path=os.path.join(result_dir, f"hu_lineprof_vol_{vol_idx}_midROI.png"),
                p0=p0, p1=p1, n=400,
                title=f"HU Line Profile (mid-slice ROI) | vol={vol_idx}"
            )

            # =============================
            # Per-slice metrics + saving
            # =============================
            for z in range(D):
                ct_z = ct_np[z]
                cbct_z = cbct_np[z]
                synth_z = synth_np[z]

                # ---- centre crop to 256×256 (same as training patches) ----
                H, W = ct_z.shape
                crop_h, crop_w = 256, 256
                h0 = (H - crop_h) // 2
                w0 = (W - crop_w) // 2

                ct_roi = ct_z[h0:h0 + crop_h, w0:w0 + crop_w]
                cbct_roi = cbct_z[h0:h0 + crop_h, w0:w0 + crop_w]
                synth_roi = synth_z[h0:h0 + crop_h, w0:w0 + crop_w]

                # ---- HU → [0,1] normalization on the ROI ----
                ct_norm = np.clip((ct_roi - HU_MIN) / (HU_MAX - HU_MIN), 0, 1)
                cbct_norm = np.clip((cbct_roi - HU_MIN) / (HU_MAX - HU_MIN), 0, 1)
                synth_norm = np.clip((synth_roi - HU_MIN) / (HU_MAX - HU_MIN), 0, 1)

                # ---- CBCT vs CT (ROI) ----
                ssim_cbct = ssim(ct_norm, cbct_norm, data_range=1)

                psnr_cbct = psnr(ct_norm, cbct_norm, data_range=1)
                mse_cbct = np.mean((ct_norm - cbct_norm) ** 2)
                mae_cbct = np.mean(np.abs(ct_norm - cbct_norm))

                # ---- Synth vs CT (ROI) ----
                ssim_synth = ssim(ct_norm, synth_norm, data_range=1)

                psnr_synth = psnr(ct_norm, synth_norm, data_range=1)
                mse_synth = np.mean((ct_norm - synth_norm) ** 2)
                mae_synth = np.mean(np.abs(ct_norm - synth_norm))

                metrics_log.append(
                    (
                        global_slice_idx,
                        ssim_cbct, psnr_cbct, mse_cbct, mae_cbct,
                        ssim_synth, psnr_synth, mse_synth, mae_synth,
                    )
                )
                global_slice_idx += 1

                # ------------------------------------------------------------------
                # Save output images for each slice
                # ------------------------------------------------------------------
                slice_id = global_slice_idx

                cbct_img = cbct_z
                ct_img = ct_z
                synth_img = synth_z

                diff_map = np.abs(ct_norm - synth_norm)

                plt.imsave(os.path.join(result_dir, f"cbct_{slice_id}.png"),
                           cbct_img, cmap="gray", vmin=vmin, vmax=vmax)
                plt.imsave(os.path.join(result_dir, f"ct_{slice_id}.png"),
                           ct_img, cmap="gray", vmin=vmin, vmax=vmax)
                plt.imsave(os.path.join(result_dir, f"synth_{slice_id}.png"),
                           synth_img, cmap="gray", vmin=vmin, vmax=vmax)
                plt.imsave(os.path.join(result_dir, f"diff_{slice_id}.png"),
                           diff_map, cmap="hot", vmin=0, vmax=1)

                fig, axes = plt.subplots(1, 3, figsize=(12, 5))
                for ax, img, title in zip(
                        axes,
                        [cbct_img, ct_img, synth_img],
                        ["CBCT Input", "Ground Truth CT", "Synthetic CT"],
                ):
                    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
                    ax.set_title(title)
                    ax.axis("off")
                plt.tight_layout()
                plt.savefig(os.path.join(result_dir, f"test_sample_{slice_id}.png"))
                plt.close(fig)

                global_slice_idx += 1

    # ============================================================
    # Save logs + averages + mean±std
    # ============================================================
    metrics_log = np.asarray(metrics_log, dtype=np.float64)
    metrics_path = os.path.join(result_dir, "test_metrics_log.txt")

    header = ("Sample\tSSIM_CBCT\tPSNR_CBCT\tMSE_CBCT\tMAE_CBCT\t"
              "SSIM_SYNTH\tPSNR_SYNTH\tMSE_SYNTH\tMAE_SYNTH")
    np.savetxt(
        metrics_path,
        metrics_log,
        fmt=['%d', '%.4f', '%.2f', '%.6f', '%.6f', '%.4f', '%.2f', '%.6f', '%.6f'],
        header=header,
        comments='',
    )

    n = metrics_log.shape[0]
    means = metrics_log[:, 1:].mean(axis=0) if n > 0 else np.zeros(8)
    stds = metrics_log[:, 1:].std(axis=0, ddof=1) if n > 1 else np.zeros(8)

    print("\n===== DATASET SUMMARY (across all slices, 3D test) =====")
    print(f"slices: {n}")
    print(f"CBCT→CT   SSIM: {means[0]:.4f} ± {stds[0]:.4f}")
    print(f"CBCT→CT   PSNR: {means[1]:.2f} dB ± {stds[1]:.2f}")
    print(f"CBCT→CT    MSE: {means[2]:.6f} ± {stds[2]:.6f}")
    print(f"CBCT→CT    MAE: {means[3]:.6f} ± {stds[3]:.6f}")
    print(f"Synth→CT  SSIM: {means[4]:.4f} ± {stds[4]:.4f}")
    print(f"Synth→CT  PSNR: {means[5]:.2f} dB ± {stds[5]:.2f}")
    print(f"Synth→CT   MSE: {means[6]:.6f} ± {stds[6]:.6f}")
    print(f"Synth→CT   MAE: {means[7]:.6f} ± {stds[7]:.6f}")

    avg_path = os.path.join(result_dir, "test_metrics_avg.txt")
    with open(avg_path, "w", encoding="utf-8") as f:
        f.write("Average Metrics Across All Slices (3D test)\n")
        f.write(f"slices: {n}\n\n")
        f.write(f"SSIM_CBCT:  {means[0]:.4f}\n")
        f.write(f"PSNR_CBCT:  {means[1]:.2f}\n")
        f.write(f"MSE_CBCT:   {means[2]:.6f}\n")
        f.write(f"MAE_CBCT:   {means[3]:.6f}\n")
        f.write(f"SSIM_SYNTH: {means[4]:.4f}\n")
        f.write(f"PSNR_SYNTH: {means[5]:.2f}\n")
        f.write(f"MSE_SYNTH:  {means[6]:.6f}\n")
        f.write(f"MAE_SYNTH:  {means[7]:.6f}\n")

    summary_path = os.path.join(result_dir, "test_metrics_mean_std.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Mean ± Std Across All Slices (3D test)\n")
        f.write(f"slices: {n}\n\n")
        f.write("CBCT→CT\n")
        f.write(f"  SSIM: {means[0]:.4f} ± {stds[0]:.4f}\n")
        f.write(f"  PSNR: {means[1]:.2f} dB ± {stds[1]:.2f}\n")
        f.write(f"  MSE : {means[2]:.6f} ± {stds[2]:.6f}\n")
        f.write(f"  MAE : {means[3]:.6f} ± {stds[3]:.6f}\n\n")
        f.write("Synth→CT\n")
        f.write(f"  SSIM: {means[4]:.4f} ± {stds[4]:.4f}\n")
        f.write(f"  PSNR: {means[5]:.2f} dB ± {stds[5]:.2f}\n")
        f.write(f"  MSE : {means[6]:.6f} ± {stds[6]:.6f}\n")
        f.write(f"  MAE : {means[7]:.6f} ± {stds[7]:.6f}\n")

    print(f"Saved per-slice metrics → {metrics_path}")
    print(f"Saved averages → {avg_path}")
    print(f"Saved mean±std → {summary_path}")
    print("Done! All metrics and visual results saved to:", result_dir)

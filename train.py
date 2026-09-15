# Copyright 2026 Alzahra Altalib, University of Dundee
# (Alzahra Altalib) email: 2600129@dundee.ac.uk
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""
Training pipeline for EPC-3D-Diff, an Equivariant Physics Consistent
Conditional 3D Latent Diffusion framework for volumetric CBCT-to-CT synthesis.

The method combines a lightweight 3D latent autoencoder, a conditional 3D
diffusion model, image-domain structural losses, and a projection-domain
rotational equivariance constraint derived from the CT acquisition operator.
Physics-based operators are used during training; inference is performed from
CBCT conditioning using the learned latent diffusion prior. [AA2026]

Reference
---------
[AA2026] A. Altalib, C. Li, H. A. Alewaidat, K. Z. Alawneh,
A. A. Qandeel, and A. Perelli,
"EPC-3D-Diff: Equivariant Physics Consistent Conditional 3D Latent Diffusion
for CBCT to CT Synthesis," arXiv:2605.20470, 2026.
Accepted to the MIART Workshop at MICCAI 2026.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from pytorch_msssim import ssim as ssim_loss
import matplotlib.pyplot as plt
from diffusion_condition_3d import ConditionalUNet as ConditionalUNet3D
from datasets import (BalancedBatchSampler, MedicalImageDataset_JUST_DICOM,
                                     MedicalImageDataset_NWH_DICOM, image_denormalization)
from src.misc import fractional_angle_shift, gradient_3d, laplacian_3d, rotate_volume_z
from ops.astracd_op import ConebeamAstraOp
from scipy.io import loadmat
from ops.astra_autograd import tigre_to_astra
from ops.Elekta_ASTRA_Geom import Elekta_geom
import astra

RAWDC_GPU = 1          # use GPU1 run ASTRA on GPU1
astra.set_gpu_index(RAWDC_GPU)

#--------------------------------------------------------------------------------------------
def _print_stats(name, t):
    t = t.detach()
    mn, mx = torch.aminmax(t)  # official PyTorch op
    mu = torch.mean(t.float())
    print(f"[rawDC-stats] {name}: min={mn.item():.4g} max={mx.item():.4g} mean={mu.item():.4g}")



def mat_path_for_patient(patient_id: str, domain_id) -> str:
    """
    patient_id: 'NWH001', 'NWH002', ... 'NWH008'
    returns full path to: Brain_raw/NWH00X/CBCTX/CBCT_NHW_scan.mat
    """
    # last character: '1'..'8'
    idx = str(patient_id)[-1]          # make sure it's a string
    cbct_folder = f"CT{idx}"           # CBCT1, CBCT2, ...

    if domain_id == 0:
        RAWDC_ROOT = NWH_raw_path
        mat_path = os.path.join(
            RAWDC_ROOT,
            patient_id,
            cbct_folder,
            "CT_NHW_scan.mat",
        )

    elif domain_id == 1:
        RAWDC_ROOT = JUST_raw_path
        mat_path = []

    return mat_path



def pretrain_latent_ae(dataloader, device, epochs=5, lr=1e-4):
    # freeze diffusion UNet
    if isinstance(model, nn.DataParallel):
        unet_params = model.module.parameters()
    else:
        unet_params = model.parameters()
    for p in unet_params:
        p.requires_grad = False

    latent_encoder.train()
    latent_decoder.train()

    opt = torch.optim.Adam(
        list(latent_encoder.parameters()) + list(latent_decoder.parameters()),
        lr=lr
    )

    for ep in range(epochs):
        total = 0.0
        for cbct, ct, pid in dataloader:
            ct = ct.to(device)
            D, H, W = ct.shape[2:]

            z = encode_latent(ct)
            ct_rec = decode_latent(z, out_size=(D, H, W))

            l1 = F.l1_loss(ct_rec, ct)
            edge = F.l1_loss(gradient_3d(ct_rec), gradient_3d(ct))
            loss = l1 + 0.2 * edge

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()

        print(f"[AE pretrain] epoch {ep+1}/{epochs}  loss={total/len(dataloader):.6f}")

    # unfreeze UNet again
    for p in unet_params:
        p.requires_grad = True


# -------------------------------------------------------------------
# Latent encoder / decoder (3D) – lightweight, no learnable params #
# -------------------------------------------------------------------
# These will be assigned in main()
latent_encoder = None
latent_decoder = None

def encode_latent(x):
    assert latent_encoder is not None, "latent_encoder has not been initialised"
    enc = latent_encoder.module if isinstance(latent_encoder, nn.DataParallel) else latent_encoder
    return enc(x)

def decode_latent(z, out_size):
    assert latent_decoder is not None, "latent_decoder has not been initialised"
    dec = latent_decoder.module if isinstance(latent_decoder, nn.DataParallel) else latent_decoder
    return dec(z, out_size=out_size)


# -------------------------------------------------------------------
# Lightweight 3D Latent Encoder / Decoder (learned, but memory-friendly)
# -------------------------------------------------------------------

class Latent3DEncoder(nn.Module):
    """
    Lightweight 3D encoder:
      - input  : [B, 1, D, H, W]
      - output : [B, LATENT_C, D, H/2, W/2]
      - downsamples only H and W (keeps depth D)
    """
    def __init__(self, in_channels: int = 1, hidden_channels: int = 16, latent_channels: int = 4):
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
                 hidden_channels: int = 64,
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


#------------------------------------
def ae_blur_test(dataloader, device, save_path="ae_test.png"):
    """
    Quick AE test:
      CT -> encoder -> decoder -> CT_recon
    Saves a 3-panel figure and prints SSIM/PSNR for the middle slice.
    """
    latent_encoder.eval()
    latent_decoder.eval()

    with torch.no_grad():  # avoids grad memory; correct for inference

        it = iter(dataloader)
        for _ in range(3):  # skip a few batches to stabilize randomness if shuffle=True
            cbct, ct, patient_id, domain_id = next(it)
        ct = ct.to(device)

        # Encode + decode ONLY (no diffusion)
        z = encode_latent(ct)
        ct_rec = decode_latent(z, out_size=ct.shape[-3:])  # (D,H,W)

        # Take middle slice for quick visual
        d = ct.shape[2] // 2

        ct_slice     = image_denormalization(ct[0, 0, d].detach().cpu().numpy())
        ct_rec_slice = image_denormalization(ct_rec[0, 0, d].detach().cpu().numpy())
        diff = abs(ct_slice - ct_rec_slice)

        # Quick metrics on that slice
        dr = float(ct_slice.max() - ct_slice.min() + 1e-8)
        s = ssim(ct_slice, ct_rec_slice, data_range=dr)
        p = psnr(ct_slice, ct_rec_slice, data_range=dr)

        print(f"[AE TEST] patient={patient_id[0]} | slice={d} | SSIM={s:.4f} | PSNR={p:.2f} dB")

        # Plot
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1); plt.title("GT CT");      plt.imshow(ct_slice, cmap="gray");     plt.axis("off")
        plt.subplot(1, 3, 2); plt.title("AE Recon");   plt.imshow(ct_rec_slice, cmap="gray"); plt.axis("off")
        plt.subplot(1, 3, 3); plt.title("|Diff|");     plt.imshow(diff, cmap="gray");         plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

        print("CT min/max:", ct.min().item(), ct.max().item())
        print("Recon min/max:", ct_rec.min().item(), ct_rec.max().item())

    # (Optional) switch back if you will continue training afterward
    latent_encoder.train()
    latent_decoder.train()


# Utilities function for AE Pre-training
def get_module(m):
    # works even if you ever wrap something in DataParallel
    return m.module if isinstance(m, nn.DataParallel) else m

def save_ae(path):
    ckpt = {
        "latent_encoder": get_module(latent_encoder).state_dict(),
        "latent_decoder": get_module(latent_decoder).state_dict(),
    }
    torch.save(ckpt, path)

def load_ae(path, device):
    ckpt = torch.load(path, map_location=device)
    get_module(latent_encoder).load_state_dict(ckpt["latent_encoder"])
    get_module(latent_decoder).load_state_dict(ckpt["latent_decoder"])


# ============================================================
#  Projection-domain equivariance loss:
#  Enforce || A(R x_sCT) - Shift( A x_CT ) ||^2 in raw projection space
# ============================================================
def raw_equivariance_loss(
    sCT: torch.Tensor,
    ct: torch.Tensor,
    patient_id_batch,
    domain_id,
    n_rot: int = 1,
    max_angles: int = 32,
    down_factor: int = 4,
    max_rotation_deg: float = 30.0,
) -> torch.Tensor:
    """
    Enforces: A(R_phi x)(θ) ≈ A(x)(θ - phi)
    sCT: [B,1,D,H,W]
    """

    device = sCT.device
    B = sCT.size(0)

    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for b in range(B):
            pid = str(patient_id_batch[b])

            if domain_id[b] == 0:             # Ninewells
                md = loadmat(mat_path_for_patient(pid, domain_id[b]))
                # --- angles ---
                angles = None
                for k in ("angles", "Angles", "theta"):
                    if k in md:
                        angles = md[k]
                        break
                if angles is None:
                    raise KeyError("Angles not found in MAT")

                # --- geometry ---
                proj_geom, _ = tigre_to_astra(md, angles)

            elif domain_id[b] == 1:        # JUST
                geom_xml_path     = "ops/Elekta_geometry.xml"
                proj_geom, angles = Elekta_geom(geom_xml_path)


            angles  = torch.tensor(np.array(angles).squeeze(),
                                  device=device, dtype=torch.float32)

            # uniform spacing assumption (required)
            delta_theta = torch.mean(angles[1:] - angles[:-1])
            Na_full     = angles.numel()

            _, _, D, H, W = sCT[b:b+1].shape
            vol_geom = astra.create_vol_geom(H, W, D)

            op = ConebeamAstraOp(proj_geom, vol_geom, device=device)

            # Convert ct [-1, 1] --> [0, 1]
            ct_n = ( ct[b:b+1] + 1.0 ) / 2.0
            # --- base projection ---
            y_full = op.A(ct_n)  # [1, Na, Hdet, Wdet]              #  A(x_ct)

            # --- subsample angles ---
            if Na_full > max_angles:
                idx = torch.sort(
                    torch.randperm(Na_full, device=device)[:max_angles]
                ).values
                y_full = y_full[:, idx]
            else:
                idx = None

            # --- detector crop ---
            _, _, Hdet, Wdet = y_full.shape
            side = min(Hdet, Wdet)
            h0 = (Hdet - side) // 2
            w0 = (Wdet - side) // 2
            y_base = y_full[..., h0:h0+side, w0:w0+side]

            # --- downsample detector ---
            if down_factor > 1:
                y_base = F.avg_pool2d(
                    y_base, kernel_size=down_factor, stride=down_factor
                )

            for _ in range(n_rot):

                # random rotation angle
                phi = (2 * torch.rand(1, device=device) - 1.0) \
                        * max_rotation_deg * np.pi / 180.0

                # Convert ct [-1, 1] --> [0, 1]
                sCT_n = (sCT[b:b+1] + 1.0) / 2.0

                # rotate volume around z
                x_rot = rotate_volume_z(sCT_n, phi)

                # project rotated volume
                y_rot = op.A(x_rot)                                                     #  A(R x_sCT)

                if idx is not None:
                    y_rot = y_rot[:, idx]

                y_rot = y_rot[..., h0:h0+side, w0:w0+side]

                if down_factor > 1:
                    y_rot = F.avg_pool2d(
                        y_rot, kernel_size=down_factor, stride=down_factor
                    )

                # convert rotation to fractional angle shift
                shift = phi / delta_theta

                # reference by interpolating base projections
                y_ref = fractional_angle_shift(y_base, shift)                       #  shift( A(x_ct) )

                total_loss += F.mse_loss(y_rot, y_ref)
                count += 1

            del op
            torch.cuda.empty_cache()

    return total_loss / max(count, 1)


# ============================================================
# Raw data consistency || y - Ax_CT ||^2
# ============================================================
def rawDC_loss(
    sCT: torch.Tensor,
    patient_id_batch,
    domain_id,
) -> torch.Tensor:
    """
    Enforces: A(R_phi x)(θ) ≈ A(x)(θ - phi)
    sCT: [B,1,D,H,W]
    """
    device = sCT.device
    B = sCT.size(0)

    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for b in range(B):
            pid = str(patient_id_batch[b])

            if domain_id[b] == 0:             # Ninewells
                md = loadmat(mat_path_for_patient(pid, domain_id[b]))
                # --- angles ---
                angles = None
                for k in ("angles", "Angles", "theta"):
                    if k in md:
                        angles = md[k]
                        break
                if angles is None:
                    raise KeyError("Angles not found in MAT")

                # --- geometry ---
                proj_geom, _ = tigre_to_astra(md, angles)

            elif domain_id[b] == 1:        # JUST
                geom_xml_path     = "ops/Elekta_geometry.xml"
                proj_geom, angles = Elekta_geom(geom_xml_path)

            angles  = torch.tensor(np.array(angles).squeeze(),
                                  device=device, dtype=torch.float32)

            Na_full = angles.numel()

            _, _, D, H, W = sCT[b:b+1].shape
            vol_geom = astra.create_vol_geom(H, W, D)

            op = ConebeamAstraOp(proj_geom, vol_geom, device=device)

            # Convert ct [-1, 1] --> [0, 1]
            sCT_n = ( sCT[b:b+1] + 1.0 ) / 2.0

            # --- base projection ---
            y_pred = op.A(sCT_n)  # [1, Na, Hdet, Wdet]              #  A(x_sCT)
            y_raw = md['proj']

            y_raw = torch.from_numpy(y_raw)

            y_raw = (
                y_raw.permute(2, 0, 1)  # HWC -> CHW
                .unsqueeze(0)  # add batch
                .contiguous()
                .to(y_pred.device)  # move to GPU
                .to(y_pred.dtype)  # match dtype
            )

            # --- subsample angles ---
            max_angles = 128
            idx = torch.sort(
                torch.randperm(Na_full, device=device)[:max_angles]
            ).values
            y_raw  = y_raw[:, idx]
            y_pred = y_pred[:, idx]

            # --- detector crop ---
            _, _, Hdet, Wdet = y_raw.shape
            side = min(Hdet, Wdet)
            h0 = (Hdet - side) // 2
            w0 = (Wdet - side) // 2
            y_raw = y_raw[..., h0:h0 + side, w0:w0 + side]
            y_pred = y_pred[..., h0:h0 + side, w0:w0 + side]

            # --- downsample detector ---
            down_factor = 4
            if down_factor > 1:
                y_raw = F.avg_pool2d(
                    y_raw, kernel_size=down_factor, stride=down_factor
                )

                y_pred = F.avg_pool2d(
                    y_pred, kernel_size=down_factor, stride=down_factor
                )

            total_loss += F.mse_loss(y_raw, y_pred)
            count += 1

            del op
            torch.cuda.empty_cache()

    return total_loss / max(count, 1)



# -------------------------------------------------------------------
# Noise scheduler (dimension-agnostic: works for 2D or 3D)
# -------------------------------------------------------------------
class NoiseScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.005):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start, t, noise=None):
        """
        x_start: (B, 1, D, H, W) or (B, 1, H, W)
        t:       (B,)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        ac = self.alphas_cumprod.to(t.device)[t]  # (B,)

        # Broadcast to match x_start dimensionality (4D or 5D)
        n_dims = x_start.dim()        # e.g. 5 for (B,1,D,H,W)
        shape = (-1,) + (1,) * (n_dims - 1)
        sqrt_ac = ac.sqrt().view(shape)
        sqrt_om = (1.0 - ac).sqrt().view(shape)

        return sqrt_ac * x_start + sqrt_om * noise



# -------------------------------------------------------------------
# 3D DDPM reverse step and sampler
# -------------------------------------------------------------------

def p_sample(model, x, t, condition, noise_scheduler):
    """
    One reverse DDPM step in latent space.

    model:      ConditionalUNet3D
    x:          [B, 1, D', H', W']  - current noisy LATENT volume
    t:          [B] (long)          - timestep indices
    condition:  [B, 1, D', H', W']  - CBCT conditioning LATENT volume
    """
    device = x.device

    betas = noise_scheduler.betas.to(device)
    alphas = noise_scheduler.alphas.to(device)
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)

    beta_t = betas[t].view(-1, 1, 1, 1, 1)
    sqrt_one_minus_ac = (1.0 - alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1, 1)
    sqrt_recip_alpha = (1.0 / alphas[t].sqrt()).view(-1, 1, 1, 1, 1)

    eps_theta = model(x, t, condition)

    model_mean = sqrt_recip_alpha * (x - beta_t / sqrt_one_minus_ac * eps_theta)

    if (t == 0).all():
        return model_mean
    else:
        noise = torch.randn_like(x)
        posterior_var = beta_t
        return model_mean + torch.sqrt(posterior_var) * noise


#------------------------------------------------

@torch.no_grad()
def sample_ddpm(model, condition, noise_scheduler, img_size=None):
    """
    3D DDPM sampler in LATENT space.

    condition: [B,1,D,H,W] (CBCT, image space)
    returns:   [B,1,D,H,W] (Synth CT, image space)
    """
    model.eval()
    device = condition.device
    B, C, D, H, W = condition.shape

    # --- encode condition to latent ---
    cond_lat = encode_latent(condition)          # [B,1,D',H',W']

    # start from pure latent noise
    x = torch.randn_like(cond_lat, device=device)

    for step in reversed(range(noise_scheduler.timesteps)):
        t_batch = torch.full((B,), step, device=device, dtype=torch.long)
        x = p_sample(model, x, t_batch, cond_lat, noise_scheduler)

    # --- decode latent back to image ---
    x_img = decode_latent(x, out_size=(D, H, W))  # [B,1,D,H,W]
    return x_img



# Add a DDIM sampler for more stable synth CT
@torch.no_grad()
def sample_ddim(model, condition, noise_scheduler, num_steps=100):
    """
    3D-safe DDIM sampler in LATENT space.

    condition: [B, 1, D, H, W] (same normalization as training, image space)
    returns:   [B, 1, D, H, W] (image space)
    """
    model.eval()
    device = condition.device
    B, _, D, H, W = condition.shape

    # encode CBCT to latent
    cond_lat = encode_latent(condition)             # [B,1,D',H',W']
    x = torch.randn_like(cond_lat, device=device)   # latent noise

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


# -------------------------------------------------------------------
# Timestep embedding
# -------------------------------------------------------------------
def get_timestep_embedding(timesteps, dim):
    """
    timesteps: (B,) long
    dim: embedding dimension
    """
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1))
    return emb



# -------------------------------------------------------------------
# Volume evaluation (SSIM/PSNR/MSE/MAE over full 3D volumes)
# -------------------------------------------------------------------
def evaluate_volume(model, dataloader, noise_scheduler, device, max_batches=2):

    model.eval()
    total_ssim_synth = 0
    total_psnr_synth = 0
    total_mse_synth = 0
    total_mae_synth = 0

    total_ssim_cbct = 0
    total_psnr_cbct = 0
    total_mse_cbct = 0
    total_mae_cbct = 0

    count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(
                tqdm(dataloader, desc="Evaluating SSIM/PSNR (3D)")
        ):
            if max_batches is not None and batch_idx >= max_batches:
                break

            # batch can be (cbct, ct, patient_id) or (cbct, ct, mask, patient_id)
            cbct_batch = batch[0]
            ct_batch = batch[1]
            # we ignore any extra items (mask, patient_id)

            cbct_batch = cbct_batch.to(device)
            ct_batch = ct_batch.to(device)

            # CHANGED (3D): use DDIM with fewer steps for faster eval
            synth_batch = sample_ddim(model, cbct_batch, noise_scheduler, num_steps=100)

            for i in range(cbct_batch.size(0)):
                # volumes: [1,D,H,W] → squeeze → [D,H,W]
                ct_vol = image_denormalization(ct_batch[i].cpu().squeeze())
                cbct_vol = image_denormalization(cbct_batch[i].cpu().squeeze())
                synth_vol = image_denormalization(synth_batch[i].cpu().squeeze())

                HU_MIN, HU_MAX = -1000, 2000
                ct_norm = np.clip((ct_vol - HU_MIN) / (HU_MAX - HU_MIN), 0, 1)
                synth_norm = np.clip((synth_vol - HU_MIN) / (HU_MAX - HU_MIN), 0, 1)
                cbct_norm = np.clip((cbct_vol - HU_MIN) / (HU_MAX - HU_MIN), 0, 1)

                # These now operate on full volumes (3D arrays)
                total_ssim_synth += ssim(ct_norm, synth_norm, data_range=1)
                total_psnr_synth += psnr(ct_norm, synth_norm, data_range=1)
                total_mse_synth += np.mean((ct_norm - synth_norm) ** 2)
                total_mae_synth += np.mean(np.abs(ct_norm - synth_norm))

                total_ssim_cbct += ssim(ct_norm, cbct_norm, data_range=1)
                total_psnr_cbct += psnr(ct_norm, cbct_norm, data_range=1)
                total_mse_cbct += np.mean((ct_norm - cbct_norm) ** 2)
                total_mae_cbct += np.mean(np.abs(ct_norm - cbct_norm))

                count += 1

    avg_ssim_synth = total_ssim_synth / count
    avg_psnr_synth = total_psnr_synth / count
    avg_mse_synth  = total_mse_synth / count
    avg_mae_synth  = total_mae_synth / count

    avg_ssim_cbct = total_ssim_cbct / count
    avg_psnr_cbct = total_psnr_cbct / count
    avg_mse_cbct = total_mse_cbct / count
    avg_mae_cbct = total_mae_cbct / count

    return (
        avg_ssim_synth,
        avg_ssim_cbct,
        avg_psnr_synth,
        avg_psnr_cbct,
        avg_mse_synth,
        avg_mse_cbct,
        avg_mae_synth,
        avg_mae_cbct,
    )



# -------------------------------------------------------------------
# Training loop with your full checkpoint + metrics logic, adapted to 3D
# -------------------------------------------------------------------
def train(model, dataloader, eval_loader, optimizer, noise_scheduler, device,
          num_epochs                = 10,
          start_epoch               = 0,
          ckpt_dir                  = "checkpoints",
          lambda_rawdc              = 1.0,
          rawdc_every_epochs        = 50,
          lambda_equiv_raw: float   = 1.0,       # base weight for equiv loss
          n_equiv_rot: int          = 1,         # rotations per sample
          equiv_every_epochs: int   = 20,        # apply equiv loss every N epochs
          equiv_target_ratio: float = 0.10       # aim: equiv term ≈ 10% of main loss
          ):

    ckpt_dir = os.path.abspath(ckpt_dir)

    os.makedirs(ckpt_dir, exist_ok=True)

    loss_log_path = os.path.join(ckpt_dir, "loss.txt")
    all_losses_log_path = os.path.join(ckpt_dir, "all_losses.txt")

    if not os.path.exists(loss_log_path):
        with open(loss_log_path, "w") as f:
            f.write("Epoch\tLoss\n")

    if not os.path.exists(all_losses_log_path):
        with open(all_losses_log_path, "w") as f:
            f.write("Epoch\tAvg_main\tAvg_diff\tAvg_L1\tAvg_edge\tAvg_lap\tAvg_raw\tLambda_raw\tAvg_equiv_loss\tLambda_equiv\n")

    printed_once = False

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0

        ######### adding zeros for each loss term ##################
        sum_main = 0.0
        sum_diff = 0.0
        sum_l1 = 0.0
        sum_edge = 0.0
        sum_lap = 0.0
        sum_raw = 0.0
        sum_equiv= 0.0
        count = 0

        ###############################################################

        rawdc_warmup_epochs = 50  # ramp-in duration (you can tune 20–100)
        use_rawdc_this_epoch = (
                lambda_rawdc > 0.0 and
                ((epoch + 1) % rawdc_every_epochs == 0)
        )

        # ---- decide if this epoch will use equivariance ----
        use_equiv_this_epoch = (
                lambda_equiv_raw > 0.0 and
                equiv_every_epochs > 0 and
                ((epoch + 1) % equiv_every_epochs == 0)
        )

        # lambda_raw_epoch is computed every epoch nu used only when use_rawdc_this epoch is enabled
        # progress = (epoch - start_epoch + 1) / float(rawdc_warmup_epochs)
        # progress = max(0.0, min(1.0, progress))
        # lambda_rawdc_epoch = lambda_rawdc * progress

        if use_rawdc_this_epoch:
            print(f"[DEBUG] rawDC ACTIVE in epoch {epoch + 1}")
        else:
            print(f"[DEBUG] rawDC SKIPPED in epoch {epoch + 1}")

        if use_equiv_this_epoch:
            print(f"[DEBUG] Equivariance ACTIVE in epoch {epoch + 1}")
        else:
            print(f"[DEBUG] Equivariance SKIPPED in epoch {epoch + 1}")


        pbar = tqdm(dataloader, desc=f"[3D] Epoch {epoch+1}")

        for cbct, ct, patient_id, domain_id in pbar:
            cbct = cbct.to(device)          # [B,1,D,H,W]
            ct   = ct.to(device)            # [B,1,D,H,W]

            patient_id_batch = patient_id  # <--- new line

            B = ct.size(0)
            t = torch.randint(0, noise_scheduler.timesteps, (B,), device=device).long()

            # -------- latent encoding (AE fixed) --------
            with torch.no_grad():
                ct_lat   = encode_latent(ct)        # [B,1,D',H',W']
                cbct_lat = encode_latent(cbct)      # [B,1,D',H',W']

            # make absolutely sure treated as constant
            ct_lat = ct_lat.detach()
            cbct_lat = cbct_lat.detach()

            noise = torch.randn_like(ct_lat)
            x_t   = noise_scheduler.q_sample(ct_lat, t, noise)

            # debug: find any param/buffer not on cuda:0
            for n, p in model.named_parameters():
                if p.is_cuda and p.device.index != 0:
                    print("PARAM ON WRONG GPU:", n, p.device)
                    break
            for n, b in model.named_buffers():
                if b.is_cuda and b.device.index != 0:
                    print("BUFFER ON WRONG GPU:", n, b.device)
                    break

            # -------- predict noise in latent space --------
            pred_noise = model(x_t, t, cbct_lat)

            # -------- diffusion loss (latent) --------
            # CHANGED: use only MSE for stability (SSIM-on-noise is noisy)
            mse = F.mse_loss(pred_noise, noise)
            try:
                ssim_term = 1 - ssim_loss(pred_noise, noise, data_range=2.0)
            except Exception:
                ssim_term = 0.0

            diffusion_loss = mse          # <-- CHANGED (before: 0.5*mse + 0.5*ssim_term)

            # -------- reconstruct x0 (latent → image) --------
            alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
            ac = alphas_cumprod[t].view(-1, 1, 1, 1, 1)  # [B,1,1,1,1]

            # DDPM formula for x0 in latent space
            x0_lat = (x_t - torch.sqrt(1.0 - ac) * pred_noise) / torch.sqrt(ac)

            if not printed_once:
                print("x0_lat device:", x0_lat.device, "shape:", tuple(x0_lat.shape))
                print("decoder type:", type(latent_decoder))
                printed_once = True

            # decode to image space for image loss (WITH gradient)
            D, H, W = ct.shape[2:]
            x0_img_for_loss = decode_latent(x0_lat, out_size=(D, H, W))  # no .detach()

            # -------- image L1 loss --------
            img_l1 = F.l1_loss(x0_img_for_loss, ct)

            # -------- edge / gradient loss to sharpen structures --------
            grad_pred = gradient_3d(x0_img_for_loss)
            grad_gt = gradient_3d(ct)
            edge_loss = F.l1_loss(grad_pred, grad_gt)

            # -------- Laplacian loss to penalise blur --------
            lap_pred = laplacian_3d(x0_img_for_loss)
            lap_gt = laplacian_3d(ct)
            lap_loss = F.l1_loss(lap_pred, lap_gt)

            # intermediate tensors BEFORE rawDC
            del grad_pred, grad_gt, lap_pred, lap_gt
            torch.cuda.empty_cache()

            # -------- main loss --------
            main_loss = (
                    0.6 * diffusion_loss
                    + 0.2 * img_l1
                    + 0.2 * edge_loss
                    + 0.1 * lap_loss
            )

            # -------- projection-data consistency loss + adaptive weight ----
            rawDC_loss_val = torch.tensor(0.0, device=ct.device)
            rawDC_weight = torch.tensor(0.0, device=ct.device)

            if use_rawdc_this_epoch and lambda_rawdc > 0.0:
                rawDC_loss_val = rawDC_loss(
                    x0_img_for_loss,  # NO detach, keep on GPU
                    patient_id_batch,
                    domain_id,
                )

                # compute the adaptive weight as in your original code …
                if rawDC_loss_val.item() > 0:
                    with torch.no_grad():
                        scale = (main_loss.detach() /
                                 (rawDC_loss_val.detach() + 1e-8)) * equiv_target_ratio
                    rawDC_weight = lambda_rawdc * scale

            # -------- projection-equivariance loss + adaptive weight ----
            equiv_raw_loss_val = torch.tensor(0.0, device=ct.device)
            equiv_weight       = torch.tensor(0.0, device=ct.device)

            if use_equiv_this_epoch and lambda_equiv_raw > 0.0:
                equiv_raw_loss_val = raw_equivariance_loss(
                    x0_img_for_loss,
                    ct,
                    patient_id_batch,
                    domain_id,
                    n_rot            = n_equiv_rot,
                    max_angles       = 32,
                    down_factor      = 4,
                    max_rotation_deg = 30.0
                )

                # compute the adaptive weight as in your original code …
                if equiv_raw_loss_val.item() > 0:
                    with torch.no_grad():
                        scale = (main_loss.detach() /
                                 (equiv_raw_loss_val.detach() + 1e-8)) * equiv_target_ratio
                    equiv_weight = lambda_equiv_raw * scale

            # final total loss remains the same
            loss = main_loss + equiv_weight * equiv_raw_loss_val + rawDC_weight * rawDC_loss_val

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            ################### save each loss ################
            sum_main  += float(main_loss.item())
            sum_diff  += float(diffusion_loss.item())
            sum_l1    += float(img_l1.item())
            sum_edge  += float(edge_loss.item())
            sum_lap   += float(lap_loss.item())
            sum_raw   += float(rawDC_loss_val.item())  # or 0.0 if rawdc not applied this iter
            sum_equiv += float(equiv_raw_loss_val.item())
            count     += 1
            ######################################################

            pbar.set_postfix({"loss": f"{loss.item():.2e}"})

        avg_loss = epoch_loss / len(dataloader)
        print(f"[3D] Epoch {epoch+1}  avg loss = {avg_loss:.6f}")

        # log loss
        with open(loss_log_path, "a") as f:
           f.write(f"{epoch+1}\t{avg_loss:.6f}\n")

        #################### save all loss terms #############
        avg_main  = sum_main / count
        avg_diff  = sum_diff / count
        avg_l1    = sum_l1 / count
        avg_edge  = sum_edge / count
        avg_lap   = sum_lap / count
        avg_raw   = sum_raw / count
        avg_equiv = sum_equiv / count

        with open(all_losses_log_path, "a") as f:
            f.write(
                f"{epoch + 1}\t{avg_main:.4f}\t{avg_diff:.4f}\t{avg_l1:.4f}\t"
                f"{avg_edge:.4f}\t{avg_lap:.4f}\t{avg_raw:.6f}\t{rawDC_weight:.6f}\t{avg_equiv:.6f}\t{equiv_weight:.6f}\n"
            )
        #################################################################

        # Save model weights
        ckpt_path = os.path.join(ckpt_dir, f"model_epoch_{epoch + 1}.pth")

        if (epoch + 1) % 50 == 0:
            try:
                ckpt = {
                    "epoch": epoch + 1,
                    "model": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                }
                torch.save(ckpt, ckpt_path)

                print(f"[3D] Saved checkpoint: {ckpt_path}")
            except Exception as e:
                print(f"⚠️ [3D] Failed to save checkpoint at epoch {epoch + 1}: {e}")

        # ----------------------------------------------------------
        # === METRICS + VISUALIZATION (every 5 epochs) ===
        # ----------------------------------------------------------
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                cbct_sample, ct_sample, patient_id_sample, domain_id_sample = next(iter(eval_loader))

                cbct_sample = cbct_sample.to(device)
                ct_sample   = ct_sample.to(device)

                # use DDIM sampler for faster, more stable sampling
                synth = sample_ddim(
                    model, cbct_sample, noise_scheduler, num_steps=100
                )

                # === Sanity Check ===
                ct_vol    = image_denormalization(ct_sample[0].cpu().squeeze().numpy())
                synth_vol = image_denormalization(synth[0].cpu().squeeze().numpy())

                if ct_vol.ndim == 3:
                    D = ct_vol.shape[0]
                    mid = D // 2
                    ct_img = ct_vol[mid]
                    synth_img = synth_vol[mid]
                else:
                    ct_img = ct_vol
                    synth_img = synth_vol

                if np.allclose(ct_img, synth_img, atol=1e-3):
                    print("Warning: Synthesized CT is nearly identical to the real CT!")
                else:
                    print("Synth CT and Real CT are different — metrics likely valid.")

                # === Visual Comparison for Debug ===
                cbct_vol = image_denormalization(cbct_sample[0].cpu().squeeze().numpy())
                if cbct_vol.ndim == 3:
                    cbct_img = cbct_vol[mid]
                else:
                    cbct_img = cbct_vol

                fig, axs = plt.subplots(1, 3, figsize=(12, 5))

                axs[0].imshow(cbct_img, cmap='gray', vmin=vmin, vmax=vmax)
                axs[0].set_title("CBCT Input", fontsize=14, fontweight='bold', y=1.05)

                axs[1].imshow(ct_img, cmap='gray', vmin=vmin, vmax=vmax)
                axs[1].set_title("Ground Truth CT", fontsize=14, fontweight='bold', y=1.05)

                axs[2].imshow(synth_img, cmap='gray', vmin=vmin, vmax=vmax)
                axs[2].set_title("Synthesized CT", fontsize=14, fontweight='bold', y=1.05)

                for ax in axs:
                    ax.axis('off')

                plt.tight_layout()
                os.makedirs(ckpt_dir, exist_ok=True)
                plt.savefig(os.path.join(ckpt_dir, f"debug_compare_{epoch + 1}.png"))
                plt.close()

                # === Save denormalized image outputs (mid-slice) ===
                HU_MIN, HU_MAX = -1000, 2000

                plt.imsave(os.path.join(ckpt_dir, f"cbct_{epoch + 1}.png"),
                           cbct_img, cmap='gray', vmin=vmin, vmax=vmax)
                plt.imsave(os.path.join(ckpt_dir, f"ct_{epoch + 1}.png"),
                           ct_img, cmap='gray', vmin=vmin, vmax=vmax)
                plt.imsave(os.path.join(ckpt_dir, f"synth_{epoch + 1}.png"),
                           synth_img, cmap='gray', vmin=vmin, vmax=vmax)

                # === Normalize for metric calculation (slice-level here) ===
                ct_norm = np.clip((ct_img - HU_MIN) / (HU_MAX - HU_MIN), 0, 1)
                synth_norm = np.clip((synth_img - HU_MIN) / (HU_MAX - HU_MIN), 0, 1)
                cbct_norm = np.clip((cbct_img - HU_MIN) / (HU_MAX - HU_MIN), 0, 1)

                # === Metrics ===
                ssim_val = ssim(ct_norm, synth_norm, data_range=1)
                psnr_val = psnr(ct_norm, synth_norm, data_range=1)
                ssim_val_original = ssim(ct_norm, cbct_norm, data_range=1)
                psnr_val_original = psnr(ct_norm, cbct_norm, data_range=1)

                mse_val = np.mean((ct_norm - synth_norm) ** 2)
                mae_val = np.mean(np.abs(ct_norm - synth_norm))

                mse_val_original = np.mean((ct_norm - cbct_norm) ** 2)
                mae_val_original = np.mean(np.abs(ct_norm - cbct_norm))

                print(
                    f"Original CT vs Synth CT → SSIM: {ssim_val:.4f}, "
                    f"PSNR: {psnr_val:.2f} dB, "
                    f"MSE: {mse_val:.6f}, MAE: {mae_val:.6f}"
                )
                print(
                    f"Original CBCT vs CT     → SSIM: {ssim_val_original:.4f}, "
                    f"PSNR: {psnr_val_original:.2f}, "
                    f"MSE: {mse_val_original:.6f}, MAE: {mae_val_original:.6f}"
                )

                # === Save to metrics_log.txt ===
                metrics_path = os.path.join(ckpt_dir, "metrics_log.txt")

                if not os.path.exists(metrics_path):
                    with open(metrics_path, "w") as f:
                        f.write("Epoch\tMSE_CBCT\tMSE_Synth\tMAE_CBCT\tMAE_Synth\t"
                                "SSIM_CBCT\tSSIM_Synth\tPSNR_CBCT\tPSNR_Synth\n")

                with open(metrics_path, "a") as f:
                    f.write(f"{epoch + 1:<6}\t"
                            f"{mse_val_original:.6f}\t{mse_val:.6f}\t"
                            f"{mae_val_original:.6f}\t{mae_val:.6f}\t"
                            f"{ssim_val_original:.4f}\t{ssim_val:.4f}\t"
                            f"{psnr_val_original:.2f}\t{psnr_val:.2f}\n")

                # === Save diff map (slice-level) ===
                diff = np.abs(ct_norm - synth_norm)
                plt.imsave(os.path.join(ckpt_dir, f"diff_{epoch + 1}.png"),
                           diff, cmap='hot')

                # === Volume-level evaluation ===
                (
                    avg_ssim_synth, avg_ssim_cbct,
                    avg_psnr_synth, avg_psnr_cbct,
                    avg_mse_synth, avg_mse_cbct,
                    avg_mae_synth, avg_mae_cbct
                ) = evaluate_volume(
                    model, eval_loader, noise_scheduler, device, max_batches=2
                )

                print(f"[Epoch {epoch + 1}] 📊 Avg Volume Metrics:")
                print(
                    f"  Original CT vs Synth CT → SSIM: {avg_ssim_synth:.4f}, "
                    f"PSNR: {avg_psnr_synth:.2f} dB, "
                    f"MSE: {avg_mse_synth:.6f}, MAE: {avg_mae_synth:.6f}"
                )
                print(
                    f"  Original CBCT vs CT     → SSIM: {avg_ssim_cbct:.4f}, "
                    f"PSNR: {avg_psnr_cbct:.2f} dB, "
                    f"MSE: {avg_mse_cbct:.6f}, MAE: {avg_mae_cbct:.6f}"
                )

                volume_log_path = os.path.join(ckpt_dir, "volume_eval_log.txt")
                if not os.path.exists(volume_log_path):
                    with open(volume_log_path, "w") as f:
                        f.write("Epoch\tMSE_CBCT\tMSE_Synth\tMAE_CBCT\tMAE_Synth\t"
                                "SSIM_CBCT\tSSIM_Synth\tPSNR_CBCT\tPSNR_Synth\n")

                with open(volume_log_path, "a") as f:
                    f.write(
                        f"{epoch + 1}\t{avg_mse_cbct:.6f}\t{avg_mse_synth:.6f}\t"
                        f"{avg_mae_cbct:.6f}\t{avg_mae_synth:.6f}\t"
                        f"{avg_ssim_cbct:.4f}\t{avg_ssim_synth:.4f}\t"
                        f"{avg_psnr_cbct:.2f}\t{avg_psnr_synth:.2f}\n"
                    )

                model.train()



# Function to load dataset
def build_dataset(name):
    if name == "JUST":
        return MedicalImageDataset_JUST_DICOM(root_dir           = JUST_data_path,
                                              mode               = "train",
                                              use_3d             = True,
                                              patch_size         = (32, 256, 256),
                                              patches_per_volume = 4,
                                              random_flip_3d     = False)

    if name == "NWH":
        return MedicalImageDataset_NWH_DICOM(root_dir           = NWH_data_path,
                                             mode               = "train",
                                             use_3d             = True,
                                             patch_size         = (32, 256, 256),
                                             patches_per_volume = 4,
                                             random_flip_3d     = False)

    if name == "JUST_NWH":
        return ConcatDataset([
            build_dataset("JUST"),
            build_dataset("NWH")
        ])

    raise ValueError(f"Unknown dataset: {name}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    pretrain_epochs = 500
    diff_num_epochs = 2500
    batch_size = 2

    DATASET_SELECTED = "NWH"             # other options: JUST, JUST_NWH

    if DATASET_SELECTED in ("JUST", "JUST_NWH"):
        JUST_data_path = "../../Dataset_JUST_np"
        JUST_raw_path  = r"../../Dataset_JUST_np/Brain_dicom"
    elif DATASET_SELECTED in ("NWH", "JUST_NWH"):
        NWH_data_path = "../../Dataset_Ninewells"
        NWH_raw_path = r"../../Dataset_Ninewells/Brain_raw"

    RAWDC_DEBUG = True  # set False later to stop prints

    RUN_AE_PRETRAIN = False  # <-- set True only when you want to train AE again
    RAWDC_ON = True
    EQUIV_ON = True

    vmin, vmax = -1000, 2000

    # ====== RESUME ======
    resume_epoch = 1450 # <<< IMPORTANT for scratch run
    start_epoch  = 0

    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # force default cuda device
        device = torch.device("cuda:0")  # explicit
    else:
        device = torch.device("cpu")

    # ====== DATASET / LOADER ======
    print("Training dataset selected: {}\n".format(DATASET_SELECTED))

    train_dataset = build_dataset(DATASET_SELECTED)

    if isinstance(train_dataset, tuple):
        JUST_len = len(train_dataset.datasets[0])
        NWH_len  = len(train_dataset.datasets[1])

        batch_sampler = BalancedBatchSampler(NWH_len, JUST_len)

        dataloader = DataLoader(
            train_dataset,
            batch_sampler = batch_sampler,
            num_workers   = 0,
            pin_memory    = True,
        )
    else:
        dataloader = DataLoader(
            train_dataset,
            batch_size  = batch_size,
            shuffle     = True,
            drop_last   = True,
            num_workers = 0,
            pin_memory  = True,
        )

    eval_loader = DataLoader(
        train_dataset,
        batch_size  = 1,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = True,
    )


    # ====== MODELS ======
    # Define this near the top
    LATENT_C = 4  # or try 16 if GPU memory allows

    # When you construct the UNet
    model = ConditionalUNet3D(
        in_channels=LATENT_C,
        cond_channels=LATENT_C,
        base_channels=64,  # increase base width
        time_emb_dim=256,
    ).to(device)

    # Construct the autoencoder to match
    latent_encoder = Latent3DEncoder(
        in_channels=1,
        hidden_channels=32,
        latent_channels=LATENT_C
    ).to(device)

    latent_decoder = Latent3DDecoder(
        in_channels=LATENT_C,
        hidden_channels=32,
        out_channels=1
    ).to(device)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = model.to(device)
        model = nn.DataParallel(model, device_ids=[0, 1], output_device=0)
    else:
        print("Using a single GPU")
        model = model.to(device)

    # ====== AE CHECK (RUN ONCE) ======
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(BASE_DIR, f"checkpoints_{DATASET_SELECTED}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Pretraining AE + training diffusion
    ae_ckpt_path = os.path.join(ckpt_dir, f"ae_latentC{LATENT_C}.pth")
    if RUN_AE_PRETRAIN:
        print("[DEBUG] __file__ =", __file__)
        print("[DEBUG] cwd     =", os.getcwd())
        print("[DEBUG] AE ckpt  =", ae_ckpt_path)
        print("[DEBUG] exists?  =", os.path.exists(ae_ckpt_path))
        print("[AE] Training AE (first time or forced)...")
        pretrain_latent_ae(dataloader, device, epochs=pretrain_epochs, lr=1e-4)
        save_ae(ae_ckpt_path)
        print("[AE] Saved:", ae_ckpt_path)

        ae_blur_test(dataloader, device, save_path="ae_test_loaded.png")

        print("Program paused to check the AE reconstruction.")
        input("Press Enter to continue...")
        print("[INFO] AE reconstruction done.")

    # load AE checkpoint and train diffusion
    else:
        print("[AE] Loading saved AE:", ae_ckpt_path)
        load_ae(ae_ckpt_path, device)

        # Safety: force AE to the correct device + eval mode (redundant but robust)
        get_module(latent_encoder).to(device)
        get_module(latent_decoder).to(device)
        get_module(latent_encoder).eval()
        get_module(latent_decoder).eval()

        print("AE encoder device:", next(get_module(latent_encoder).parameters()).device)
        print("AE decoder device:", next(get_module(latent_decoder).parameters()).device)
    # ......................................

    # after AE stage, before optimizer (U-Net to train with diffusion)
    for p in get_module(model).parameters():
        p.requires_grad = True

    # Freeze AE (Fixed weights)
    get_module(latent_encoder).eval()
    get_module(latent_decoder).eval()
    for p in get_module(latent_encoder).parameters():
        p.requires_grad = False
    for p in get_module(latent_decoder).parameters():
        p.requires_grad = False

    # optional: reduce fragmentation after AE stage
    torch.cuda.empty_cache()
    #=============================================================

    # ====== OPTIMIZER / SCHEDULER ======
    scheduler = NoiseScheduler(timesteps=1000)
    unet_params = get_module(model).parameters()

    # Optimizer only based on U-Net for diffusion
    # while encoder and decoder parameters are fixed
    optimizer = torch.optim.Adam(unet_params, lr=1e-5)

    # ====== RESUME FROM CHECKPOINT (OPTIONAL) ======
    from collections import OrderedDict

    if resume_epoch is not None:
        checkpoint_path = os.path.join(ckpt_dir, f"model_epoch_{resume_epoch}.pth")
        if os.path.exists(checkpoint_path):

            ckpt = torch.load(checkpoint_path, map_location="cuda:0")

            # ---- NEW FORMAT: dict with 'model', 'latent_encoder', 'latent_decoder' ----
            if isinstance(ckpt, dict) and "model" in ckpt:
                print("Resuming from FULL checkpoint (model)")
                state_dict = ckpt["model"]

                # handle DataParallel for model
                first_key = list(state_dict.keys())[0]
                if isinstance(model, nn.DataParallel) and not first_key.startswith("module."):
                    state_dict = OrderedDict((f"module.{k}", v) for k, v in state_dict.items())
                elif not isinstance(model, nn.DataParallel) and first_key.startswith("module."):
                    state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
                model.load_state_dict(state_dict)

                start_epoch = ckpt.get("epoch", resume_epoch)

            # ---- OLD FORMAT: file is just model.state_dict() (no latent) ----
            else:
                print("Resuming from OLD checkpoint (model only, latent = random)")
                state_dict = ckpt
                first_key = list(state_dict.keys())[0]
                if isinstance(model, nn.DataParallel) and not first_key.startswith("module."):
                    state_dict = OrderedDict((f"module.{k}", v) for k, v in state_dict.items())
                elif not isinstance(model, nn.DataParallel) and first_key.startswith("module."):
                    state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
                model.load_state_dict(state_dict)
                start_epoch = resume_epoch

            print(f"Resumed from epoch {start_epoch}")
        else:
            print(f"Checkpoint not found at {checkpoint_path}. Starting from scratch.")
            start_epoch = 0

    if RAWDC_ON:
        lambda_rawdc = 1  # extra safety (even if someone sets operator later)
        rawdc_every_epochs = 10  # irrelevant when rawdc_operator_fn=None
    else:
        lambda_rawdc = 0.0  # extra safety (even if someone sets operator later)
        rawdc_every_epochs = 10

    if EQUIV_ON:
        lambda_equiv_raw = 1
        n_equiv_rot = 1
        equiv_every_epochs = 10
        equiv_target_ratio = 0.10
    else:
        lambda_equiv_raw = 0
        n_equiv_rot = 1
        equiv_every_epochs = 10
        equiv_target_ratio = 0.10

    print("UNet trainable?", any(p.requires_grad for p in get_module(model).parameters()))
    print("AE enc trainable?", any(p.requires_grad for p in get_module(latent_encoder).parameters()))
    print("AE dec trainable?", any(p.requires_grad for p in get_module(latent_decoder).parameters()))

    train(
        model=model,
        dataloader=dataloader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        noise_scheduler=scheduler,
        device=device,
        num_epochs=diff_num_epochs,
        start_epoch=start_epoch,
        ckpt_dir=ckpt_dir,
        lambda_rawdc=lambda_rawdc,
        rawdc_every_epochs=rawdc_every_epochs,
        lambda_equiv_raw=lambda_equiv_raw,
        n_equiv_rot=n_equiv_rot,
        equiv_every_epochs=equiv_every_epochs,
        equiv_target_ratio=equiv_target_ratio,
    )

    # ====== FINAL VISUALIZATION AFTER TRAINING ======
    model.eval()
    with torch.no_grad():
        cbct_sample, ct_sample, patient_id_sample, domain_id = next(iter(eval_loader))

        cbct_sample = cbct_sample.to(device)
        ct_sample = ct_sample.to(device)

        generated_ct = sample_ddpm(
            model,
            cbct_sample,
            scheduler,
            img_size=cbct_sample.shape[1:],  # kept for compatibility
        )

        # convert to numpy volumes: [1,D,H,W] → [D,H,W]
        cbct_vol = cbct_sample[0].cpu().squeeze().numpy()
        ct_vol   = ct_sample[0].cpu().squeeze().numpy()
        synth_vol = generated_ct[0].cpu().squeeze().numpy()

        # pick middle slice along depth
        if cbct_vol.ndim == 3:
            D = cbct_vol.shape[0]
            mid = D // 2
            cbct_img = cbct_vol[mid]
            ct_img   = ct_vol[mid]
            synth_img = synth_vol[mid]
        else:
            cbct_img = cbct_vol
            ct_img   = ct_vol
            synth_img = synth_vol

        # plot 3 panels like before (but now using mid-slice)
        plt.figure(figsize=(10, 5))
        titles = ["Input CBCT (mid slice)", "Input CT (mid slice)", "Synthesized CT (mid slice)"]
        imgs = [cbct_img, ct_img, synth_img]

        for i, (img, title) in enumerate(zip(imgs, titles)):
            plt.subplot(1, 3, i + 1)
            plt.imshow(image_denormalization(img), cmap='gray', vmin=vmin, vmax=vmax)
            plt.title(title)
            plt.axis('off')

        plt.tight_layout()
        plt.show()
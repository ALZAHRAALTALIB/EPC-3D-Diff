# Copyright 2026 Alzahra Altalib, University of Dundee
# (Alzahra Altalib) email: 2600129@dundee.ac.uk
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""
Conditional 3D U-Net used as the latent denoising network in EPC-3D-Diff.

The network receives a noisy CT latent representation, a diffusion timestep,
and the paired CBCT latent condition, and predicts the injected diffusion
noise using 3D residual encoder-decoder blocks. [AA2026]

Reference
---------
[AA2026] A. Altalib, C. Li, H. A. Alewaidat, K. Z. Alawneh,
A. A. Qandeel, and A. Perelli,
"EPC-3D-Diff: Equivariant Physics Consistent Conditional 3D Latent Diffusion
for CBCT to CT Synthesis," arXiv:2605.20470, 2026.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Time Embedding ----------

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        timesteps: (B,) integer or float
        returns: (B, dim)
        """
        device = timesteps.device
        half_dim = self.dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        exponents = torch.exp(torch.arange(half_dim, device=device) * -emb_factor)
        emb = timesteps.float().unsqueeze(1) * exponents.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            # if odd dim, pad one extra zero
            emb = F.pad(emb, (0, 1))
        return emb


# ---------- Basic 3D Blocks ----------

class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()

        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels)
            )
        else:
            self.time_mlp = None

        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t_emb=None):
        """
        x: (B, C, D, H, W)
        t_emb: (B, time_emb_dim) or None
        """
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        if t_emb is not None and self.time_mlp is not None:
            temb = self.time_mlp(t_emb)  # (B, C_out)
            # reshape to (B, C_out, 1, 1, 1) for broadcasting
            temb = temb[:, :, None, None, None]
            h = h + temb

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return h + self.skip(x)


class DownBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.res1 = ResBlock3D(in_channels, out_channels, time_emb_dim)
        self.res2 = ResBlock3D(out_channels, out_channels, time_emb_dim)
        self.down = nn.Conv3d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, t_emb):
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock3D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, time_emb_dim):
        super().__init__()
        # upsample from in_channels -> out_channels
        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )

        # after up, we concatenate with skip: channels = out_channels + skip_channels
        self.res1 = ResBlock3D(out_channels + skip_channels, out_channels, time_emb_dim)
        self.res2 = ResBlock3D(out_channels, out_channels, time_emb_dim)

    def forward(self, x, skip, t_emb):
        # x:   (B, C_in, D, H, W)
        # skip:(B, C_skip, D_skip, H_skip, W_skip)
        x = self.up(x)

        # shape alignment if needed
        if x.shape[-3:] != skip.shape[-3:]:
            diff_d = skip.shape[-3] - x.shape[-3]
            diff_h = skip.shape[-2] - x.shape[-2]
            diff_w = skip.shape[-1] - x.shape[-1]

            x = F.pad(
                x,
                (
                    diff_w // 2, diff_w - diff_w // 2,
                    diff_h // 2, diff_h - diff_h // 2,
                    diff_d // 2, diff_d - diff_d // 2,
                )
            )

        # concatenate along channels
        x = torch.cat([x, skip], dim=1)  # channels = out_channels + skip_channels
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        return x




# ---------- Main 3D Conditional UNet ----------

class ConditionalUNet(nn.Module):
    """
    3D Conditional UNet for diffusion.

    Expected usage:
        model = ConditionalUNet(in_channels=1,
                                cond_channels=1,
                                base_channels=64,
                                time_emb_dim=256)

        out = model(x, t, cond)

    where:
        x:     (B, 1, D, H, W) noisy CT (or CBCT)
        cond:  (B, 1, D, H, W) conditioning image (CBCT or CT)
        t:     (B,) timesteps
        out:   (B, 1, D, H, W) predicted noise
    """

    def __init__(self, in_channels=1, cond_channels=1, base_channels=64, time_emb_dim=256):
        super().__init__()

        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.time_emb_dim = time_emb_dim

        # time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # we concatenate input and condition along channels
        input_channels = in_channels + cond_channels

        # Encoder
        self.init_conv = nn.Conv3d(input_channels, base_channels, kernel_size=3, padding=1)

        self.down1 = DownBlock3D(base_channels, base_channels * 2, time_emb_dim)
        self.down2 = DownBlock3D(base_channels * 2, base_channels * 4, time_emb_dim)
        self.down3 = DownBlock3D(base_channels * 4, base_channels * 8, time_emb_dim)

        # Bottleneck
        self.mid1 = ResBlock3D(base_channels * 8, base_channels * 8, time_emb_dim)
        self.mid2 = ResBlock3D(base_channels * 8, base_channels * 8, time_emb_dim)


        # Decoder
        # Decoder in ConditionalUNet
        self.up3 = UpBlock3D(
            in_channels=base_channels * 8,  # from bottleneck
            skip_channels=base_channels * 8,  # s3
            out_channels=base_channels * 4,
            time_emb_dim=time_emb_dim,
        )
        self.up2 = UpBlock3D(
            in_channels=base_channels * 4,  # from up3
            skip_channels=base_channels * 4,  # s2
            out_channels=base_channels * 2,
            time_emb_dim=time_emb_dim,
        )
        self.up1 = UpBlock3D(
            in_channels=base_channels * 2,  # from up2
            skip_channels=base_channels * 2,  # s1
            out_channels=base_channels,
            time_emb_dim=time_emb_dim,
        )

        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv3d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t, cond):
        """
        x:    (B, C_in, D, H, W)
        cond: (B, C_cond, D, H, W)
        t:    (B,)
        """
        # concatenate input + condition
        h = torch.cat([x, cond], dim=1)

        # time embedding
        t_emb = self.time_mlp(t)  # (B, time_emb_dim)

        # encoder
        h = self.init_conv(h)
        d1, s1 = self.down1(h, t_emb)  # d1: downsampled, s1: skip
        d2, s2 = self.down2(d1, t_emb)
        d3, s3 = self.down3(d2, t_emb)

        # bottleneck
        m = self.mid1(d3, t_emb)
        m = self.mid2(m, t_emb)

        # decoder
        u3 = self.up3(m, s3, t_emb)
        u2 = self.up2(u3, s2, t_emb)
        u1 = self.up1(u2, s1, t_emb)

        out = self.out_norm(u1)
        out = self.out_act(out)
        out = self.out_conv(out)

        return out


if __name__ == "__main__":
    # Quick shape test (run: python diffusion_condition_3d.py)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalUNet(
        in_channels=1,
        cond_channels=1,
        base_channels=32,   # use 32 for test to save memory
        time_emb_dim=256
    ).to(device)

    B, D, H, W = 1, 32, 64, 64  # small volume / patch
    x = torch.randn(B, 1, D, H, W, device=device)
    cond = torch.randn(B, 1, D, H, W, device=device)
    t = torch.randint(0, 1000, (B,), device=device)

    with torch.no_grad():
        y = model(x, t, cond)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)

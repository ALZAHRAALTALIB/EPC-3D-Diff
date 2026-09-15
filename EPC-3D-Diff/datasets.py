# Copyright 2026 Alzahra Altalib, University of Dundee
# (Alzahra Altalib) email: 2600129@dundee.ac.uk
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""
Dataset loading and preprocessing utilities for EPC-3D-Diff.

The loaders support paired 3D CBCT/CT preparation, slice-stack alignment,
in-plane registration, HU normalization, fixed-size preprocessing, and random
3D patch sampling for the NWH phantom and JUST clinical cohorts used in the
EPC-3D-Diff study. [AA2026]

Reference
---------
[AA2026] A. Altalib, C. Li, H. A. Alewaidat, K. Z. Alawneh,
A. A. Qandeel, and A. Perelli,
"EPC-3D-Diff: Equivariant Physics Consistent Conditional 3D Latent Diffusion
for CBCT to CT Synthesis," arXiv:2605.20470, 2026.
"""


import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import re
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
import pydicom
from scipy.io import loadmat
from scipy.ndimage import shift as nd_shift
from skimage.registration import phase_cross_correlation
from skimage.metrics import structural_similarity as ssim
from skimage.filters import sobel
import random  # for random 3D patch sampling

# ---------------- Configuration ----------------
TARGET_HEIGHT = 256
TARGET_WIDTH  = 256
HU_MIN = -1000.0
HU_MAX = 2000.0
FORCE_SLICES = 92            # enforce exactly 92 pairs
PROFILE_PERCENTILE = 60      # robust profile percentile
QC_SAMPLES = 8               # used by external checker
# ------------------------------------------------

# ---------------- HU helpers ----------------
def normalize_fixed_window(img2d):
    x = np.clip(img2d, HU_MIN, HU_MAX)
    return 2.0 * (x - HU_MIN) / (HU_MAX - HU_MIN) - 1.0

def image_denormalization(nimg):
    if isinstance(nimg, torch.Tensor):
        nimg = nimg.cpu().numpy()
    return (((nimg + 1) / 2.0) * (HU_MAX - HU_MIN) + HU_MIN).astype(np.float32)

def scale_to_hu_linear(arr):
    a, b = float(np.min(arr)), float(np.max(arr))
    if b <= a + 1e-9:
        return np.full_like(arr, HU_MIN, dtype=np.float32)
    return HU_MIN + (arr - a) * (HU_MAX - HU_MIN) / (b - a)

# --------------- misc utils ----------------
def _glob_dicoms_recursive(path):
    files = []
    for dp, _, fnames in os.walk(path):
        for f in fnames:
            files.append(os.path.join(dp, f))
    return sorted(files)

def _find_series_dir_strict(pdir: str, kind: str):
    kids = [k for k in os.listdir(pdir) if os.path.isdir(os.path.join(pdir, k))]
    rx = re.compile(rf"^{kind}(\d+)?$", re.IGNORECASE)
    cands = [k for k in kids if rx.match(k)]
    if not cands:
        return None
    best, best_n = None, -1
    for k in cands:
        full = os.path.join(pdir, k)
        n = len(_glob_dicoms_recursive(full))
        if n > best_n:
            best, best_n = full, n
    return best if best_n > 0 else None

# --------------- DICOM & TIGRE loaders ---------------
def load_dicom_volume(folder_path, flip=False):
    slices = []
    for f in sorted(os.listdir(folder_path)):
        p = os.path.join(folder_path, f)
        try:
            ds = pydicom.dcmread(p, force=True)
            if getattr(ds, "PixelData", None) is not None:
                slices.append(ds)
        except Exception:
            continue
    if not slices:
        raise FileNotFoundError(f"No DICOMs under: {folder_path}")

    def zpos(d):
        if hasattr(d, "ImagePositionPatient"):
            try:
                return float(d.ImagePositionPatient[2])
            except Exception:
                pass
        return float(getattr(d, "InstanceNumber", 0))

    slices.sort(key=zpos)
    vol = np.stack([s.pixel_array for s in slices]).astype(np.float32)
    slope = float(getattr(slices[0], "RescaleSlope", 1.0))
    intercept = float(getattr(slices[0], "RescaleIntercept", 0.0))
    vol = vol * slope + intercept
    if flip:
        vol = vol[::-1]
    return vol
#------------------------------
def _load_tigre_fdk_mat(folder_path):
    """
    Loader for TIGRE FDK recon volumes in Ninewells dataset.

    Assumptions (true for your data):
      - In each CBCT folder (e.g. .../NWH001/CBCT1) there is exactly one file:
            CBCT_NHW_scan.mat
      - That MAT file contains the FDK volume under the key 'img_FDK'
      - The volume is 3D (or nested in an extra leading dimension) and can be
        coerced to shape (Z, Y, X).

    Supports both MATLAB v5 and v7.3/HDF5.
    """
    import os
    import numpy as np
    from scipy.io import loadmat

    try:
        import h5py
    except Exception:
        h5py = None

    mpath = os.path.join(folder_path, "CBCT_NHW_scan.mat")
    if not os.path.exists(mpath):
        raise FileNotFoundError(f"CBCT_NHW_scan.mat not found under {folder_path}")

    def _as_volume(arr):
        arr = np.array(arr)
        # peel off singleton dims until <= 3D
        while arr.ndim > 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 3 or arr.size < 1000:
            return None

        # Try to ensure (Z, Y, X) ordering.
        # For Ninewells FDK we typically have something like (93, 512, 512).
        # If first dim is smallest, we already have Z,Y,X.
        if arr.shape[0] <= min(arr.shape[1], arr.shape[2]):
            vol = arr
        else:
            # otherwise, put the smallest axis first as Z
            z_axis = int(np.argmin(arr.shape))
            axes = [z_axis] + [i for i in range(3) if i != z_axis]
            vol = arr.transpose(axes)
        return vol.astype(np.float32)

    # --- try MATLAB v5 reader first ---
    vol = None
    try:
        md = loadmat(mpath, squeeze_me=True, struct_as_record=False)
        if "img_FDK" in md:
            vol = _as_volume(md["img_FDK"])
    except Exception:
        vol = None

    # --- if that failed and file is v7.3, try h5py ---
    if (vol is None) and (h5py is not None):
        try:
            with h5py.File(mpath, "r") as f:
                if "img_FDK" in f:
                    vol = _as_volume(f["img_FDK"][()])
        except Exception:
            vol = None

    if vol is None:
        raise RuntimeError(
            f"Could not extract a valid 3D 'img_FDK' volume from {mpath}"
        )

    print(f"[FDK] Loaded volume from: {mpath}  shape={vol.shape}")
    return vol

#------------------------------

def crop_image_inplane(volume, thr=-300):
    mask2d = (np.max(volume, axis=0) > thr)
    if not mask2d.any():
        return volume
    ys, xs = np.where(mask2d)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    return volume[:, y0:y1+1, x0:x1+1]

def crop_image_inplane_paired(ct_vol, cbct_vol, thr=-300, margin=0, pad_val=-1000.0):
    """
    Compute ONE (y0:y1, x0:x1) bbox from the UNION mask of CT and CBCT,
    then crop BOTH volumes with the same bbox.
    Keeps ct_vol and cbct_vol in-plane shapes identical.
    """
    # masks from max-projection
    m_ct   = (np.max(ct_vol,   axis=0) > thr)
    m_cbct = (np.max(cbct_vol, axis=0) > thr)
    mask = (m_ct | m_cbct)

    if not mask.any():
        return ct_vol, cbct_vol

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    # optional margin
    y0 = max(0, y0 - margin); x0 = max(0, x0 - margin)
    y1 = min(mask.shape[0] - 1, y1 + margin)
    x1 = min(mask.shape[1] - 1, x1 + margin)

    # crop both
    ct_c   = ct_vol[:,   y0:y1+1, x0:x1+1]
    cbct_c = cbct_vol[:, y0:y1+1, x0:x1+1]
    return ct_c, cbct_c

# --------------- orientation + z pairing ---------------
def _slice_profile_percentile(vol, p=PROFILE_PERCENTILE):
    thr = np.percentile(vol, p)
    mask = (vol > thr).astype(np.float32)
    prof = mask.sum(axis=(1,2))
    return (prof - prof.mean()) / (prof.std() + 1e-6)

def _best_shift_1d(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    cc = np.correlate(a, b, mode="full")
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
    lag = int(np.argmax(cc) - (len(b) - 1))
    return lag, float(cc.max()/denom)

def _build_idx_map(cb_len, ct_len, shift, reversed_ct):
    base = (ct_len - 1 - np.arange(cb_len)) if reversed_ct else np.arange(cb_len)
    idx = base + shift
    valid = (idx >= 0) & (idx < ct_len)
    if not np.any(valid):
        return [], None
    return idx[valid].astype(int).tolist(), np.where(valid)[0]

def build_idx_map_by_profile(cbct_vol, ct_vol, domain_id):
    prof_c = _slice_profile_percentile(cbct_vol)
    prof_t = _slice_profile_percentile(ct_vol)
    s_same, r_same = _best_shift_1d(prof_c, prof_t)
    s_rev,  r_rev  = _best_shift_1d(prof_c, prof_t[::-1])

    if (r_rev, -abs(s_rev)) > (r_same, -abs(s_same)):
        use_rev, shift, corr = True, s_rev, r_rev
    else:
        use_rev, shift, corr = False, s_same, r_same

    if domain_id == 1:
        use_rev = False
    idx_map, keep = _build_idx_map(cbct_vol.shape[0], ct_vol.shape[0], shift, use_rev)
    cbct_kept = cbct_vol[keep] if keep is not None else cbct_vol[:0]
    info = {"use_rev": use_rev, "shift": int(shift), "corr": float(corr)}
    return idx_map, cbct_kept, info

# --------------- per-patient (dy,dx) ---------------
def _estimate_patient_xy_shift(cbct_vol, ct_vol, idx_map, nsamples=16):
    if not idx_map:
        return 0.0, 0.0
    K = len(idx_map)
    start = max(2, K//10); end = max(start+1, K - K//10)
    num = min(nsamples, end - start)
    picks = np.linspace(start, end-1, num=num, dtype=int)
    dys, dxs = [], []
    for k in picks:
        A = cbct_vol[k].astype(np.float32)
        B = ct_vol[idx_map[k]].astype(np.float32)
        A0 = (A - np.median(A)) / (np.std(A) + 1e-6)
        B0 = (B - np.median(B)) / (np.std(B) + 1e-6)
        # downscale for stability if large
        target = 192
        h,w = A0.shape; scale = 1.0
        if max(h,w) > target:
            s = target / max(h,w)
            nh,nw = max(1,int(round(h*s))), max(1,int(round(w*s)))
            A0 = F.interpolate(torch.from_numpy(A0)[None,None], size=(nh,nw),
                               mode="bilinear", align_corners=False).squeeze().numpy()
            B0 = F.interpolate(torch.from_numpy(B0)[None,None], size=(nh,nw),
                               mode="bilinear", align_corners=False).squeeze().numpy()
            scale = 1.0/s
        (dy, dx), _, _ = phase_cross_correlation(A0, B0, upsample_factor=10)
        dys.append(float(dy)*scale); dxs.append(float(dx)*scale)
    return float(np.median(dys)), float(np.median(dxs))

# --------- ROI edge refinement + apply-if-better ----------
def _central_roi(img, frac=0.70):
    H, W = img.shape
    h, w = int(H*frac), int(W*frac)
    y0 = (H - h)//2; x0 = (W - w)//2
    return img[y0:y0+h, x0:x0+w]

def _pad_to_same(a, b, val=-1000.0):
    """Center-pad two 2D arrays to the same (H,W)."""
    Ha, Wa = a.shape
    Hb, Wb = b.shape
    H, W = max(Ha, Hb), max(Wa, Wb)

    def pad(x):
        out = np.full((H, W), val, dtype=np.float32)
        y0 = (H - x.shape[0]) // 2
        x0 = (W - x.shape[1]) // 2
        out[y0:y0 + x.shape[0], x0:x0 + x.shape[1]] = x
        return out

    if (Ha, Wa) != (H, W):
        a = pad(a)
    if (Hb, Wb) != (H, W):
        b = pad(b)
    return a, b

def refine_shift_edges(cb, ct, dy0, dx0, frac=0.70, up=50):
    """Refine (dy,dx) using Sobel edges on a central ROI; tolerant to size mismatch."""
    from scipy.ndimage import shift as imshift

    # 1) pad to same size first
    cb, ct = _pad_to_same(cb.astype(np.float32), ct.astype(np.float32), val=-1000.0)

    # 2) apply coarse shift to CT
    ct0 = imshift(ct, shift=(dy0, dx0), order=1, mode='nearest')

    # 3) central ROI to avoid pads/couch
    cb_r = _central_roi(cb, frac=frac)
    ct_r = _central_roi(ct0, frac=frac)

    # 4) edges + normalization
    cb_e = sobel(cb_r); ct_e = sobel(ct_r)
    cb_n = (cb_e - np.median(cb_e)) / (np.std(cb_e) + 1e-6)
    ct_n = (ct_e - np.median(ct_e)) / (np.std(ct_e) + 1e-6)

    # 5) subpixel phase correlation
    (ddy, ddx), _, _ = phase_cross_correlation(cb_n, ct_n, upsample_factor=up)
    return float(dy0 + ddy), float(dx0 + ddx)

def should_apply_shift(cb, ct, dy, dx, frac=0.70):
    """Decide whether to apply (dy,dx) based on ROI-SSIM improvement; size-agnostic."""
    from scipy.ndimage import shift as imshift

    # pad to same size first
    cb, ct = _pad_to_same(cb.astype(np.float32), ct.astype(np.float32), val=-1000.0)

    # ROI before
    cb_b, ct_b = _central_roi(cb, frac), _central_roi(ct, frac)
    s_b = ssim((cb_b - cb_b.mean())/(cb_b.std()+1e-6),
               (ct_b - ct_b.mean())/(ct_b.std()+1e-6),
               data_range=2.0)

    # ROI after shift
    ct_s = imshift(ct, shift=(dy, dx), order=1, mode='nearest')
    cb_a, ct_a = _central_roi(cb, frac), _central_roi(ct_s, frac)
    s_a = ssim((cb_a - cb_a.mean())/(cb_a.std()+1e-6),
               (ct_a - ct_a.mean())/(ct_a.std()+1e-6),
               data_range=2.0)

    gain = s_a - s_b
    return (gain > 0.01), s_b, s_a


# --------------- pairwise crop & letterbox ---------------
def letterbox_to_fixed_size(img, H, W):
    h, w = img.shape
    if h == 0 or w == 0:
        return np.zeros((H,W), dtype=np.float32)
    scale = min(H/h, W/w)
    nh, nw = max(1,int(round(h*scale))), max(1,int(round(w*scale)))
    t = torch.from_numpy(img).float()[None,None]
    t = F.interpolate(t, size=(nh,nw), mode="bilinear", align_corners=False)
    resized = t.squeeze().numpy()
    canvas = np.full((H,W), HU_MIN, dtype=resized.dtype)
    top = (H-nh)//2; left = (W-nw)//2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas

def pairwise_trim(ct, cbct, thr=-950, margin=6):
    """
    Safe paired crop that tolerates different in-plane sizes.
    If shapes differ, both are padded to the larger (H,W) first.
    """
    if ct.shape != cbct.shape:
        Ha, Wa = ct.shape
        Hb, Wb = cbct.shape
        Ht, Wt = max(Ha, Hb), max(Wa, Wb)

        def pad_center(img, H, W, val=-1000.0):
            out = np.full((H, W), val, dtype=np.float32)
            y0 = (H - img.shape[0]) // 2
            x0 = (W - img.shape[1]) // 2
            out[y0:y0+img.shape[0], x0:x0+img.shape[1]] = img
            return out

        ct   = pad_center(ct,   Ht, Wt)
        cbct = pad_center(cbct, Ht, Wt)

    mask = (ct > thr) | (cbct > thr)
    if not mask.any():
        return ct, cbct
    ys, xs = np.where(mask)
    y0 = max(0, ys.min()-margin)
    y1 = min(ct.shape[0]-1, ys.max()+margin)
    x0 = max(0, xs.min()-margin)
    x1 = min(ct.shape[1]-1, xs.max()+margin)
    return ct[y0:y1+1, x0:x1+1], cbct[y0:y1+1, x0:x1+1]

# ---------------- Dataset ----------------
# ---------------- Dataset ----------------
class MedicalImageDataset_JUST_DICOM(Dataset):
    """
    Expects:
      root/Brain_raw_/<PATIENT>/{CT,CBCT}/...
    - CT: DICOM series
    - CBCT: TIGRE FDK .mat volume

    2D mode (default):
      - use_3d = False  → original slice-level behaviour (one (cbct, ct) per slice)

    3D mode:
      - use_3d = True   → returns (cbct_patch, ct_patch) with shape [1, Dz, Dy, Dx]
                           using the SAME preprocessing (pairwise_trim, letterbox,
                           HU window, alignment) but stacked into a volume.
    """

    def __init__(self, root_dir, mode="train",
                 return_mask=False,
                 use_3d=False,                    #  : NEW flag
                 patch_size=(16, 256, 256),       #  : (Dz, Dy, Dx)
                 patches_per_volume=4,            #  : how many samples per patient
                 random_flip_3d=True              #  : random flips as augmentation
                 ):
        self.root_dir = os.path.abspath(root_dir)
        self.mode = (mode or "train").lower()
        self.return_mask = return_mask
        self.use_3d = use_3d                     #  : store flag
        self.patch_size = patch_size             #
        self.patches_per_volume = patches_per_volume  #
        self.random_flip_3d = random_flip_3d     #
        self.domain_id = 1

        base = os.path.join(self.root_dir, "Brain_dicom")
        patient_root = base if os.path.isdir(base) else self.root_dir

        all_patients = sorted(
            os.path.join(patient_root, d)
            for d in os.listdir(patient_root)
            if os.path.isdir(os.path.join(patient_root, d))
        )
        if not all_patients:
            raise FileNotFoundError(f"No patients under: {patient_root}")

        n = len(all_patients)
        cut = max(1, int(0.8*n)) if n>1 else 1
        if self.mode == "train":
            self.patient_dirs = all_patients[:cut]
        elif self.mode == "test":
            self.patient_dirs = all_patients[cut:] or all_patients
        else:
            self.patient_dirs = all_patients

        print(f"Loaded {len(self.patient_dirs)} patients (mode={self.mode})")
        self.data = []                            # original slice-level index
        self.volumes_3d = []                     #  : per-patient 3D volumes

        for p in self.patient_dirs:
            ct_dir   = _find_series_dir_strict(p, "CT")
            cbct_dir = _find_series_dir_strict(p, "CBCT")
            if not (ct_dir and cbct_dir):
                print(f"⚠️ Skipping {os.path.basename(p)} (missing CT/CBCT)")
                continue

            # Load
            ct_vol = np.load(f"{ct_dir}/CT.npy")
            cbct_vol = np.load(f"{cbct_dir}/CBCT.npy")
            # cbct_vol = scale_to_hu_linear(cbct_vol)   ######### CRITICAL FOR JUST

            # Robust in-plane crop (PAIRED)
            ct_vol, cbct_vol = crop_image_inplane_paired(ct_vol, cbct_vol, thr=-300, margin=0)

            # Pairing (orientation + z-shift)
            idx_map, cbct_kept, info = build_idx_map_by_profile(cbct_vol, ct_vol, self.domain_id)

            if len(idx_map) == 0:
                print(f"No overlap; skipping {os.path.basename(p)}")
                continue

            # Coarse dy,dx
            dy0, dx0 = _estimate_patient_xy_shift(cbct_kept, ct_vol, idx_map)

            # ROI edge refinement on mid slice
            mid = len(idx_map)//2
            dy1, dx1 = refine_shift_edges(cbct_kept[mid], ct_vol[idx_map[mid]],
                                          dy0, dx0, frac=0.70, up=50)

            # Apply only if ROI-SSIM improves
            apply, s_b, s_a = should_apply_shift(cbct_kept[mid], ct_vol[idx_map[mid]], dy1, dx1, frac=0.70)
            dy, dx = (dy1, dx1) if apply else (0.0, 0.0)

            print(f"→ {os.path.basename(p)}: "
                  f"{'reversed' if info['use_rev'] else 'same'}; z_shift={info['shift']}; "
                  f"corr≈{info['corr']:.3f}; pairs={len(idx_map)}; "
                  f"coarse=({dy0:.2f},{dx0:.2f}) refined=({dy1:.2f},{dx1:.2f}) "
                  f"ROI-SSIM {s_b:.3f}->{s_a:.3f} applied={apply}")

            # -------- Original slice-level entries (kept for 2D + raw-DC) --------
            self.data.append({
                "name": os.path.basename(p),
                "ct": ct_vol,
                "cbct": cbct_kept,
                "ct_index_for_cbct": [int(i) for i in idx_map],
                "num_slices": int(len(idx_map)),
                "xy_shift": (float(dy), float(dx)),
            })

            # -------- NEW: build fully preprocessed 3D volumes for this patient --------  #
            if self.use_3d:
                ct_slices = []
                cb_slices = []
                valid_slices = []
                for k, z_ct in enumerate(idx_map):
                    cb = cbct_kept[k]
                    ct = ct_vol[z_ct]

                    # apply same (dy,dx) shift as 2D path
                    if abs(dy) > 1e-2 or abs(dx) > 1e-2:
                        ct = nd_shift(ct.astype(np.float32), shift=(dy, dx),
                                      order=1, mode="nearest")

                    # same paired crop + resize
                    ct_c, cb_c = pairwise_trim(ct, cb, thr=-950, margin=6)
                    ct_res = letterbox_to_fixed_size(ct_c, TARGET_HEIGHT, TARGET_WIDTH)
                    cb_res = letterbox_to_fixed_size(cb_c, TARGET_HEIGHT, TARGET_WIDTH)

                    valid = ((ct_res > -950).astype(np.float32) *
                             (cb_res > -950).astype(np.float32))

                    ctN = normalize_fixed_window(ct_res)
                    cbN = normalize_fixed_window(cb_res)

                    ct_slices.append(ctN.astype(np.float32))
                    cb_slices.append(cbN.astype(np.float32))
                    valid_slices.append(valid.astype(np.float32))

                ct_vol3d = np.stack(ct_slices, axis=0)
                cbct_vol3d = np.stack(cb_slices, axis=0)
                valid_vol3d = np.stack(valid_slices, axis=0)

                self.volumes_3d.append({
                    "name": os.path.basename(p),
                    "ct": ct_vol3d,
                    "cbct": cbct_vol3d,
                    "valid": valid_vol3d,
                })

        if not self.data:
            raise RuntimeError("Dataset contains 0 usable patients after loading.")

    #############################################################################
    def __len__(self):
        if self.use_3d:
            if self.mode == "test":
                # One full volume per patient
                return len(self.volumes_3d)
            else:
                # Training: random patches
                return len(self.volumes_3d) * self.patches_per_volume
        return sum(d["num_slices"] for d in self.data)

    #####################################################################################

    def _sample_3d_patch(self, vol_ct, vol_cb, vol_valid):
        """Sample a random 3D patch from preprocessed volumes."""
        D, H, W = vol_ct.shape
        Dz, Dy, Dx = self.patch_size

        if Dz > D or Dy > H or Dx > W:
            raise RuntimeError(
                f"3D patch {self.patch_size} larger than volume {(D,H,W)}"
            )

        z0 = random.randint(0, D - Dz)
        y0 = random.randint(0, H - Dy)
        x0 = random.randint(0, W - Dx)

        ct_patch = vol_ct[z0:z0+Dz, y0:y0+Dy, x0:x0+Dx]
        cb_patch = vol_cb[z0:z0+Dz, y0:y0+Dy, x0:x0+Dx]
        vm_patch = vol_valid[z0:z0+Dz, y0:y0+Dy, x0:x0+Dx]

        ct_t = torch.from_numpy(ct_patch).float()
        cb_t = torch.from_numpy(cb_patch).float()
        vm_t = torch.from_numpy(vm_patch).float()

        # Optional random flips in 3D
        if self.random_flip_3d:
            if random.random() < 0.5:
                ct_t = torch.flip(ct_t, dims=[0]); cb_t = torch.flip(cb_t, dims=[0]); vm_t = torch.flip(vm_t, dims=[0])
            if random.random() < 0.5:
                ct_t = torch.flip(ct_t, dims=[1]); cb_t = torch.flip(cb_t, dims=[1]); vm_t = torch.flip(vm_t, dims=[1])
            if random.random() < 0.5:
                ct_t = torch.flip(ct_t, dims=[2]); cb_t = torch.flip(cb_t, dims=[2]); vm_t = torch.flip(vm_t, dims=[2])

        # add channel dimension: (1, Dz, Dy, Dx)
        ct_t = ct_t.unsqueeze(0)
        cb_t = cb_t.unsqueeze(0)
        vm_t = vm_t.unsqueeze(0)

        if self.return_mask:
            return cb_t, ct_t, vm_t
        return cb_t, ct_t

    def __getitem__(self, index):
        # ---------- 3D MODE ----------
        if self.use_3d:

            if not self.volumes_3d:
                raise RuntimeError("use_3d=True but no 3D volumes were built.")

            # ==========================
            # TEST MODE → RETURN FULL VOLUME
            # ==========================
            if self.mode == "test":
                rec = self.volumes_3d[index]

                print("TEST MODE FULL VOLUME SHAPE:", rec["ct"].shape)

                ct_full = torch.from_numpy(rec["ct"]).float().unsqueeze(0)
                cb_full = torch.from_numpy(rec["cbct"]).float().unsqueeze(0)

                patient_id = rec["name"]

                return cb_full, ct_full, patient_id, self.domain_id

            # ==========================
            # TRAIN MODE → RANDOM PATCH
            # ==========================
            else:
                vol_idx = index // self.patches_per_volume
                vol_idx = min(vol_idx, len(self.volumes_3d) - 1)

                rec = self.volumes_3d[vol_idx]

                cb_t, ct_t = self._sample_3d_patch(
                    rec["ct"], rec["cbct"], rec["valid"]
                )

                patient_id = rec["name"]

                return cb_t, ct_t, patient_id, self.domain_id

        # ---------- ORIGINAL 2D SLICE-LEVEL BEHAVIOUR ----------
        cum = 0
        for d in self.data:
            N = d["num_slices"]
            if index < cum + N:
                k = index - cum
                cb = d["cbct"][k]
                ct = d["ct"][d["ct_index_for_cbct"][k]]

                dy, dx = d["xy_shift"]
                if abs(dy) > 1e-2 or abs(dx) > 1e-2:
                    ct = nd_shift(ct.astype(np.float32), shift=(dy,dx), order=1, mode="nearest")

                # paired crop
                ct, cb = pairwise_trim(ct, cb, thr=-950, margin=6)

                # resize to network input
                ct_res = letterbox_to_fixed_size(ct, TARGET_HEIGHT, TARGET_WIDTH)
                cb_res = letterbox_to_fixed_size(cb, TARGET_HEIGHT, TARGET_WIDTH)

                # valid mask to ignore pads in loss
                valid = (ct_res > -950).astype(np.float32) * (cb_res > -950).astype(np.float32)

                # [-1,1] normalization
                ctN = normalize_fixed_window(ct_res)
                cbN = normalize_fixed_window(cb_res)

                ct_t = torch.from_numpy(ctN).float().unsqueeze(0)
                cb_t = torch.from_numpy(cbN).float().unsqueeze(0)
                if self.return_mask:
                    vm_t = torch.from_numpy(valid).float().unsqueeze(0)
                    return cb_t, ct_t, vm_t
                return cb_t, ct_t
            cum += N
        raise IndexError("Index out of range")


# ---------------- Dataset ----------------
class MedicalImageDataset_NWH_DICOM(Dataset):
    """
    Expects:
      root/Brain_raw_/<PATIENT>/{CT,CBCT}/...
    - CT: DICOM series
    - CBCT: TIGRE FDK .mat volume

    2D mode (default):
      - use_3d = False  → original slice-level behaviour (one (cbct, ct) per slice)

    3D mode:
      - use_3d = True   → returns (cbct_patch, ct_patch) with shape [1, Dz, Dy, Dx]
                           using the SAME preprocessing (pairwise_trim, letterbox,
                           HU window, alignment) but stacked into a volume.
    """

    def __init__(self, root_dir, mode="train",
                 return_mask=False,
                 use_3d=False,
                 patch_size=(16, 256, 256),  # (Dz, Dy, Dx)
                 patches_per_volume=4,
                 random_flip_3d=True
                 ):
        self.root_dir = os.path.abspath(root_dir)
        self.mode = (mode or "train").lower()
        self.return_mask = return_mask
        self.use_3d = use_3d
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.random_flip_3d = random_flip_3d
        self.domain_id = 0

        base = os.path.join(self.root_dir, "Brain_raw")
        patient_root = base if os.path.isdir(base) else self.root_dir

        all_patients = sorted(
            os.path.join(patient_root, d)
            for d in os.listdir(patient_root)
            if os.path.isdir(os.path.join(patient_root, d))
        )
        if not all_patients:
            raise FileNotFoundError(f"No patients under: {patient_root}")

        n = len(all_patients)
        cut = max(1, int(0.8 * n)) if n > 1 else 1
        if self.mode == "train":
            self.patient_dirs = all_patients[:cut]
        elif self.mode == "test":
            self.patient_dirs = all_patients[cut:] or all_patients
        else:
            self.patient_dirs = all_patients

        print(f"Loaded {len(self.patient_dirs)} patients (mode={self.mode})")
        self.data = []  # original slice-level index
        self.volumes_3d = []  # per-patient 3D volumes

        for p in self.patient_dirs:
            ct_dir = _find_series_dir_strict(p, "CT")
            cbct_dir = _find_series_dir_strict(p, "CBCT")
            if not (ct_dir and cbct_dir):
                print(f"Skipping {os.path.basename(p)} (missing CT/CBCT)")
                continue

            # Load
            ct_vol = load_dicom_volume(ct_dir, flip=False)
            cbct_vol = _load_tigre_fdk_mat(cbct_dir)
            cbct_vol = scale_to_hu_linear(cbct_vol)

            # Robust in-plane crop
            ct_vol = crop_image_inplane(ct_vol)
            cbct_vol = crop_image_inplane(cbct_vol)

            # Pairing (orientation + z-shift)
            idx_map, cbct_kept, info = build_idx_map_by_profile(cbct_vol, ct_vol, self.domain_id)

            # Enforce exactly 92 pairs after orientation
            if FORCE_SLICES and len(cbct_kept) != FORCE_SLICES:
                L = len(cbct_kept)
                if L > FORCE_SLICES:
                    extra = L - FORCE_SLICES
                    if info['use_rev']:
                        cbct_kept = cbct_kept[extra:];
                        idx_map = idx_map[extra:]
                    else:
                        cbct_kept = cbct_kept[:-extra];
                        idx_map = idx_map[:-extra]
                else:
                    keep = min(L, len(idx_map))
                    cbct_kept = cbct_kept[:keep];
                    idx_map = idx_map[:keep]

            if len(idx_map) == 0:
                print(f"No overlap; skipping {os.path.basename(p)}")
                continue

            # Coarse dy,dx
            dy0, dx0 = _estimate_patient_xy_shift(cbct_kept, ct_vol, idx_map)

            # ROI edge refinement on mid slice
            mid = len(idx_map) // 2
            dy1, dx1 = refine_shift_edges(cbct_kept[mid], ct_vol[idx_map[mid]],
                                          dy0, dx0, frac=0.70, up=50)

            # Apply only if ROI-SSIM improves
            apply, s_b, s_a = should_apply_shift(cbct_kept[mid], ct_vol[idx_map[mid]], dy1, dx1, frac=0.70)
            dy, dx = (dy1, dx1) if apply else (0.0, 0.0)

            print(f"→ {os.path.basename(p)}: "
                  f"{'reversed' if info['use_rev'] else 'same'}; z_shift={info['shift']}; "
                  f"corr≈{info['corr']:.3f}; pairs={len(idx_map)}; "
                  f"coarse=({dy0:.2f},{dx0:.2f}) refined=({dy1:.2f},{dx1:.2f}) "
                  f"ROI-SSIM {s_b:.3f}->{s_a:.3f} applied={apply}")

            # -------- Original slice-level entries (kept for 2D + raw-DC) --------
            self.data.append({
                "name": os.path.basename(p),
                "ct": ct_vol,
                "cbct": cbct_kept,
                "ct_index_for_cbct": [int(i) for i in idx_map],
                "num_slices": int(len(idx_map)),
                "xy_shift": (float(dy), float(dx)),
            })

            # -------- NEW: build fully preprocessed 3D volumes for this patient --------
            if self.use_3d:
                ct_slices = []
                cb_slices = []
                valid_slices = []
                for k, z_ct in enumerate(idx_map):
                    cb = cbct_kept[k]
                    ct = ct_vol[z_ct]

                    # apply same (dy,dx) shift as 2D path
                    if abs(dy) > 1e-2 or abs(dx) > 1e-2:
                        ct = nd_shift(ct.astype(np.float32), shift=(dy, dx),
                                      order=1, mode="nearest")

                    # same paired crop + resize
                    ct_c, cb_c = pairwise_trim(ct, cb, thr=-950, margin=6)
                    ct_res = letterbox_to_fixed_size(ct_c, TARGET_HEIGHT, TARGET_WIDTH)
                    cb_res = letterbox_to_fixed_size(cb_c, TARGET_HEIGHT, TARGET_WIDTH)

                    valid = ((ct_res > -950).astype(np.float32) *
                             (cb_res > -950).astype(np.float32))

                    ctN = normalize_fixed_window(ct_res)
                    cbN = normalize_fixed_window(cb_res)

                    ct_slices.append(ctN.astype(np.float32))
                    cb_slices.append(cbN.astype(np.float32))
                    valid_slices.append(valid.astype(np.float32))

                ct_vol3d = np.stack(ct_slices, axis=0)
                cbct_vol3d = np.stack(cb_slices, axis=0)
                valid_vol3d = np.stack(valid_slices, axis=0)

                self.volumes_3d.append({
                    "name": os.path.basename(p),
                    "ct": ct_vol3d,
                    "cbct": cbct_vol3d,
                    "valid": valid_vol3d,
                })

        if not self.data:
            raise RuntimeError("Dataset contains 0 usable patients after loading.")

    #############################################################################
    def __len__(self):
        if self.use_3d:
            if self.mode == "test":
                # One full volume per patient
                return len(self.volumes_3d)
            else:
                # Training: random patches
                return len(self.volumes_3d) * self.patches_per_volume
        return sum(d["num_slices"] for d in self.data)
    #####################################################################################

    def _sample_3d_patch(self, vol_ct, vol_cb, vol_valid):
        """Sample a random 3D patch from preprocessed volumes."""
        D, H, W = vol_ct.shape
        Dz, Dy, Dx = self.patch_size

        if Dz > D or Dy > H or Dx > W:
            raise RuntimeError(
                f"3D patch {self.patch_size} larger than volume {(D, H, W)}"
            )

        z0 = random.randint(0, D - Dz)
        y0 = random.randint(0, H - Dy)
        x0 = random.randint(0, W - Dx)

        ct_patch = vol_ct[z0:z0 + Dz, y0:y0 + Dy, x0:x0 + Dx]
        cb_patch = vol_cb[z0:z0 + Dz, y0:y0 + Dy, x0:x0 + Dx]
        vm_patch = vol_valid[z0:z0 + Dz, y0:y0 + Dy, x0:x0 + Dx]

        ct_t = torch.from_numpy(ct_patch).float()
        cb_t = torch.from_numpy(cb_patch).float()
        vm_t = torch.from_numpy(vm_patch).float()

        # Optional random flips in 3D
        if self.random_flip_3d:
            if random.random() < 0.5:
                ct_t = torch.flip(ct_t, dims=[0]);
                cb_t = torch.flip(cb_t, dims=[0]);
                vm_t = torch.flip(vm_t, dims=[0])
            if random.random() < 0.5:
                ct_t = torch.flip(ct_t, dims=[1]);
                cb_t = torch.flip(cb_t, dims=[1]);
                vm_t = torch.flip(vm_t, dims=[1])
            if random.random() < 0.5:
                ct_t = torch.flip(ct_t, dims=[2]);
                cb_t = torch.flip(cb_t, dims=[2]);
                vm_t = torch.flip(vm_t, dims=[2])

        # add channel dimension: (1, Dz, Dy, Dx)
        ct_t = ct_t.unsqueeze(0)
        cb_t = cb_t.unsqueeze(0)
        vm_t = vm_t.unsqueeze(0)

        if self.return_mask:
            return cb_t, ct_t, vm_t
        return cb_t, ct_t

    def __getitem__(self, index):

        # ---------- 3D MODE ----------
        if self.use_3d:

            if not self.volumes_3d:
                raise RuntimeError("use_3d=True but no 3D volumes were built.")

            # ==========================
            # TEST MODE → RETURN FULL VOLUME
            # ==========================
            if self.mode == "test":
                rec = self.volumes_3d[index]

                print("TEST MODE FULL VOLUME SHAPE:", rec["ct"].shape)

                ct_full = torch.from_numpy(rec["ct"]).float().unsqueeze(0)
                cb_full = torch.from_numpy(rec["cbct"]).float().unsqueeze(0)

                patient_id = rec["name"]

                return cb_full, ct_full, patient_id, self.domain_id

            # ==========================
            # TRAIN MODE → RANDOM PATCH
            # ==========================
            else:
                vol_idx = index // self.patches_per_volume
                vol_idx = min(vol_idx, len(self.volumes_3d) - 1)

                rec = self.volumes_3d[vol_idx]

                cb_t, ct_t = self._sample_3d_patch(
                    rec["ct"], rec["cbct"], rec["valid"]
                )

                patient_id = rec["name"]

                return cb_t, ct_t, patient_id, self.domain_id

        # ---------- ORIGINAL 2D SLICE-LEVEL BEHAVIOUR ----------
        cum = 0
        for d in self.data:
            N = d["num_slices"]
            if index < cum + N:
                k = index - cum
                cb = d["cbct"][k]
                ct = d["ct"][d["ct_index_for_cbct"][k]]

                dy, dx = d["xy_shift"]
                if abs(dy) > 1e-2 or abs(dx) > 1e-2:
                    ct = nd_shift(ct.astype(np.float32), shift=(dy, dx), order=1, mode="nearest")

                # paired crop
                ct, cb = pairwise_trim(ct, cb, thr=-950, margin=6)

                # resize to network input
                ct_res = letterbox_to_fixed_size(ct, TARGET_HEIGHT, TARGET_WIDTH)
                cb_res = letterbox_to_fixed_size(cb, TARGET_HEIGHT, TARGET_WIDTH)

                # valid mask to ignore pads in loss
                valid = (ct_res > -950).astype(np.float32) * (cb_res > -950).astype(np.float32)

                # [-1,1] normalization
                ctN = normalize_fixed_window(ct_res)
                cbN = normalize_fixed_window(cb_res)

                ct_t = torch.from_numpy(ctN).float().unsqueeze(0)
                cb_t = torch.from_numpy(cbN).float().unsqueeze(0)
                if self.return_mask:
                    vm_t = torch.from_numpy(valid).float().unsqueeze(0)
                    return cb_t, ct_t, vm_t
                return cb_t, ct_t
            cum += N
        raise IndexError("Index out of range")





class BalancedBatchSampler(Sampler):
    """
    Ensures each batch contains:
        1 phantom sample
        1 clinical sample
    """

    def __init__(self, len_phantom, len_clinical):
        self.len_phantom = len_phantom
        self.len_clinical = len_clinical
        self.offset = len_phantom  # where clinical indices start

        # oversample smaller dataset automatically
        self.num_batches = max(len_phantom, len_clinical)

    def __iter__(self):
        phantom_indices = list(range(self.len_phantom))
        clinical_indices = list(range(self.len_clinical))

        random.shuffle(phantom_indices)
        random.shuffle(clinical_indices)

        for i in range(self.num_batches):
            p_idx = phantom_indices[i % self.len_phantom]
            c_idx = clinical_indices[i % self.len_clinical] + self.offset
            yield [p_idx, c_idx]  # batch of size 2

    def __len__(self):
        return self.num_batches
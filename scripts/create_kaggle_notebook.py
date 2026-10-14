#!/usr/bin/env python3
"""
Generates MMIF-INet Kaggle Notebooks and sets up 3 independent Kaggle jobs:
1. Task 1 (Baseline): Paper setup (PET-MRI, SPECT-MRI, ks=1.0)
2. Task 2 (Job 1 with CT): 3 modalities (CT-MRI, PET-MRI, SPECT-MRI) with NO color loss (ks=0.0)
3. Task 3 (Job 2 with CT): 3 modalities with conditional color loss (ks=1.0 on PET/SPECT, 0 on CT)
"""
import nbformat as nbf
import os
import copy
import json

def build_notebook(experiment_name='job1_with_ct', per_gpu_batch=128, val_freq=100):
    nb = nbf.v4.new_notebook()
    cells = []

    # Cell 1: Markdown header
    cells.append(nbf.v4.new_markdown_cell(f"""# Multimodal Medical Image Fusion (MMIF-INet) Training
### Accelerated Multi-GPU Pipeline on Dual NVIDIA Tesla T4 GPUs (`GPU T4 x2`)
**Active Experiment Target**: `{experiment_name}`

---

### Available Work Plan Tasks:
1. **`baseline`**: Exact paper reproduction on PET-MRI and SPECT-MRI ($\\\\alpha=3.0, \\\\beta=1.0, \\\\gamma=1.0, ks=1.0$).
2. **`job1_with_ct`**: Joint 3-modality training (CT-MRI, PET-MRI, SPECT-MRI) with **NO color loss** ($ks=0.0$, anatomical & functional fusion without chrominance loss).
3. **`job2_with_ct`**: Joint 3-modality training (CT-MRI, PET-MRI, SPECT-MRI) with **conditional color loss** ($ks=1.0$, color supervision active on PET/SPECT, disabled on CT).

### Optimizations:
- **Fast In-Memory RAM Buffering**: Entire Harvard-AANLIB dataset (61 MB) buffered once into RAM in ~1.4s, eliminating 1,476,000 disk I/O & PNG decompressions.
- **Microsecond Tensor Augmentations**: Crop and flips executed directly in PyTorch memory.
- **Dual T4 Acceleration**: Multi-GPU `DataParallel` with automatic batch scaling (`BATCH_SIZE = {per_gpu_batch} * num_gpus`).
- **AMP Mixed Precision**: FP16 Tensor Core execution for high throughput.
"""))

    # Cell 2: Markdown
    cells.append(nbf.v4.new_markdown_cell("""## 1. System Environment & Dual T4 GPU Diagnostics"""))

    # Cell 3: Code
    cells.append(nbf.v4.new_code_cell("""import os
import sys
import time
import math
import glob
import random
import warnings
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from scipy.signal import correlate2d

warnings.filterwarnings('ignore')

# Dual GPU Diagnostics
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")

num_gpus = torch.cuda.device_count()
print(f"Total GPUs:      {num_gpus}")

if torch.cuda.is_available():
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024 ** 3)
        print(f"  -> [GPU {i}] {props.name} | VRAM: {vram_gb:.2f} GB | Compute: {props.major}.{props.minor}")
    primary_device = torch.device('cuda:0')
else:
    print("WARNING: Running on CPU. Ensure Kaggle accelerator is set to GPU T4 x2!")
    primary_device = torch.device('cpu')

# Reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

seed_everything(42)
print("Hardware diagnostics and seeding complete.")
"""))

    # Cell 4: Markdown
    cells.append(nbf.v4.new_markdown_cell("""## 2. Dataset Acquisition & Verification"""))

    # Cell 5: Code
    cells.append(nbf.v4.new_code_cell("""def setup_harvard_dataset():
    # 1. Search Kaggle input mounts
    kaggle_mounts = [
        '/kaggle/input/havard-medical-image-fusion-datasets',
        '/kaggle/input/harvard-medical-image-fusion-datasets',
        '/kaggle/input/harvard-aanlib',
        '/kaggle/input/harvardmedicalimagefusion'
    ]
    for mount in kaggle_mounts:
        if os.path.exists(mount) and os.path.exists(os.path.join(mount, 'MyDatasets')):
            print(f"-> Found dataset mounted at: {mount}")
            return mount

    # 2. Search local workspace directories
    candidate_dirs = [
        'dataset/Harvard-AANLIB',
        '../dataset/Harvard-AANLIB',
        '../../dataset/Harvard-AANLIB',
        'Harvard-AANLIB'
    ]
    for d in candidate_dirs:
        abs_d = os.path.abspath(d)
        if os.path.exists(abs_d) and os.path.exists(os.path.join(abs_d, 'MyDatasets')):
            print(f"-> Found local dataset directory: {abs_d}")
            return abs_d

    # 3. Clone repository automatically
    target = '/kaggle/working/Harvard-AANLIB' if os.path.exists('/kaggle/working') else './dataset/Harvard-AANLIB'
    target_abs = os.path.abspath(target)
    if not os.path.exists(target_abs):
        print(f"-> Dataset not found. Auto-cloning Havard-Medical-Image-Fusion-Datasets to {target_abs}...")
        os.makedirs(os.path.dirname(target_abs), exist_ok=True)
        clone_cmd = f"git clone --depth 1 https://github.com/xianming-gu/Havard-Medical-Image-Fusion-Datasets.git {target_abs}"
        ret = os.system(clone_cmd)
        if ret != 0:
            raise RuntimeError(f"Failed to clone dataset repository (returncode {ret}).")
        print("-> Dataset clone complete!")
    return target_abs

DATA_ROOT = setup_harvard_dataset()
print(f"Active DATA_ROOT: {DATA_ROOT}\\n")

tasks = ['PET-MRI', 'SPECT-MRI', 'CT-MRI']
mod_map = {'PET-MRI': 'PET', 'SPECT-MRI': 'SPECT', 'CT-MRI': 'CT'}
for task in tasks:
    for split in ['train', 'test']:
        mri_p = os.path.join(DATA_ROOT, 'MyDatasets', task, split, 'MRI')
        mod_p = os.path.join(DATA_ROOT, 'MyDatasets', task, split, mod_map[task])
        n_mri = len([f for f in os.listdir(mri_p) if f.lower().endswith(('.png', '.jpg'))]) if os.path.exists(mri_p) else 0
        n_mod = len([f for f in os.listdir(mod_p) if f.lower().endswith(('.png', '.jpg'))]) if os.path.exists(mod_p) else 0
        print(f"  [{task:>9} | {split:>5}]  MRI: {n_mri:3d}  |  {mod_map[task]:>5}: {n_mod:3d} pairs")
"""))

    # Cell 6: Markdown
    cells.append(nbf.v4.new_markdown_cell("""## 3. High-Speed RAM-Buffered Dataset & DataLoaders"""))

    # Cell 7: Code
    cells.append(nbf.v4.new_code_cell("""class PairedTransform:
    def __init__(self, cropsize, is_train=True):
        self.cropsize = cropsize
        self.is_train = is_train

    def __call__(self, img1, img2):
        if self.is_train:
            w, h = img1.size
            if self.cropsize is not None and w > self.cropsize and h > self.cropsize:
                i = random.randint(0, h - self.cropsize)
                j = random.randint(0, w - self.cropsize)
                img1 = TF.crop(img1, i, j, self.cropsize, self.cropsize)
                img2 = TF.crop(img2, i, j, self.cropsize, self.cropsize)

            if random.random() > 0.5:
                img1 = TF.hflip(img1)
                img2 = TF.hflip(img2)

            if random.random() > 0.5:
                img1 = TF.vflip(img1)
                img2 = TF.vflip(img2)
        else:
            if self.cropsize is not None:
                img1 = TF.center_crop(img1, (self.cropsize, self.cropsize))
                img2 = TF.center_crop(img2, (self.cropsize, self.cropsize))

        return TF.to_tensor(img1), TF.to_tensor(img2)


class MedicalFusionDataset(Dataset):
    def __init__(self, data_root, tasks=('PET-MRI', 'SPECT-MRI'), split='train', cropsize=128, return_color_flag=False, preload_to_ram=True):
        self.pairs = []
        self.return_color_flag = return_color_flag
        self.preload_to_ram = preload_to_ram
        self.cropsize = cropsize
        self.is_train = (split == 'train')
        self.transform = PairedTransform(cropsize=cropsize, is_train=self.is_train)

        task_mod_map = {
            'PET-MRI': 'PET',
            'SPECT-MRI': 'SPECT',
            'CT-MRI': 'CT'
        }

        for task in tasks:
            other_mod = task_mod_map[task]
            mri_dir = os.path.join(data_root, 'MyDatasets', task, split, 'MRI')
            other_dir = os.path.join(data_root, 'MyDatasets', task, split, other_mod)

            if not os.path.exists(mri_dir) or not os.path.exists(other_dir):
                continue

            mri_files = set(f for f in os.listdir(mri_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
            other_files = set(f for f in os.listdir(other_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
            common_files = sorted(list(mri_files & other_files))

            is_color = 0.0 if task == 'CT-MRI' else 1.0

            for fname in common_files:
                self.pairs.append((
                    os.path.join(mri_dir, fname),
                    os.path.join(other_dir, fname),
                    is_color
                ))

        # In-Memory RAM Buffer (Eliminates repeated disk I/O and PNG decompression)
        if self.preload_to_ram and len(self.pairs) > 0:
            print(f"[{split:>5}] Pre-loading {len(self.pairs)} image pairs into RAM buffer...")
            t0 = time.time()
            self.mri_buffers = []
            self.other_buffers = []
            self.color_flags = []
            for mri_p, other_p, is_col in self.pairs:
                cov = TF.to_tensor(Image.open(mri_p).convert('RGB'))
                sec = TF.to_tensor(Image.open(other_p).convert('RGB'))
                self.mri_buffers.append(cov)
                self.other_buffers.append(sec)
                self.color_flags.append(torch.tensor(is_col, dtype=torch.float32))
            elapsed = time.time() - t0
            print(f"[{split:>5}] Buffered {len(self.pairs)} pairs in RAM in {elapsed:.2f}s!")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if self.preload_to_ram:
            cover = self.mri_buffers[idx]
            secret = self.other_buffers[idx]
            is_col = self.color_flags[idx]

            # Fast microsecond tensor augmentation
            if self.is_train and self.cropsize is not None:
                _, h, w = cover.shape
                if h > self.cropsize and w > self.cropsize:
                    i = random.randint(0, h - self.cropsize)
                    j = random.randint(0, w - self.cropsize)
                    cover = cover[:, i:i+self.cropsize, j:j+self.cropsize]
                    secret = secret[:, i:i+self.cropsize, j:j+self.cropsize]

                if random.random() > 0.5:
                    cover = torch.flip(cover, dims=[2])
                    secret = torch.flip(secret, dims=[2])

                if random.random() > 0.5:
                    cover = torch.flip(cover, dims=[1])
                    secret = torch.flip(secret, dims=[1])
            else:
                if self.cropsize is not None:
                    _, h, w = cover.shape
                    if h >= self.cropsize and w >= self.cropsize:
                        i = (h - self.cropsize) // 2
                        j = (w - self.cropsize) // 2
                        cover = cover[:, i:i+self.cropsize, j:j+self.cropsize]
                        secret = secret[:, i:i+self.cropsize, j:j+self.cropsize]

            if self.return_color_flag:
                return cover, secret, is_col
            return cover, secret

        # Fallback to disk loading
        mri_path, other_path, is_color = self.pairs[idx]
        cover = Image.open(mri_path).convert('RGB')
        secret = Image.open(other_path).convert('RGB')
        data0, data1 = self.transform(cover, secret)
        if self.return_color_flag:
            return data0, data1, torch.tensor(is_color, dtype=torch.float32)
        return data0, data1


def get_dataloaders(data_root, tasks=('PET-MRI', 'SPECT-MRI'), batch_size=32, cropsize=128, cropsize_val=256, return_color_flag=False, num_workers=2):
    train_ds = MedicalFusionDataset(
        data_root=data_root,
        tasks=tasks,
        split='train',
        cropsize=cropsize,
        return_color_flag=return_color_flag,
        preload_to_ram=True
    )
    val_ds = MedicalFusionDataset(
        data_root=data_root,
        tasks=tasks,
        split='test',
        cropsize=cropsize_val,
        return_color_flag=return_color_flag,
        preload_to_ram=True
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1
    )
    return train_loader, val_loader
"""))

    # Cell 8: Markdown
    cells.append(nbf.v4.new_markdown_cell("""## 4. Wavelet Operations (Haar DWT & IWT)"""))

    # Cell 9: Code
    cells.append(nbf.v4.new_code_cell("""def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2.0
    x02 = x[:, :, 1::2, :] / 2.0
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return x_LL, x_HL, x_LH, x_HH


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_channel = int(in_channel / (r ** 2))
    out_height = r * in_height
    out_width = r * in_width

    x1 = x[:, 0:out_channel, :, :] / 2.0
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2.0
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2.0
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2.0

    h = torch.zeros([in_batch, out_channel, out_height, out_width], device=x.device, dtype=x.dtype)
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)
"""))

    # Cell 10: Markdown
    cells.append(nbf.v4.new_markdown_cell("""## 5. Residual Dense Subnetworks (RRDB) & Invertible Coupling Blocks"""))

    # Cell 11: Code
    cells.append(nbf.v4.new_code_cell("""def initialize_weights(net_l, scale=1.0):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias.data, 0.0)


class ResidualDenseBlock_out(nn.Module):
    def __init__(self, input_c, output_c, bias=True):
        super(ResidualDenseBlock_out, self).__init__()
        self.conv1 = nn.Conv2d(input_c, 32, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(input_c + 32, 32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input_c + 2 * 32, 32, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(input_c + 3 * 32, 32, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(input_c + 4 * 32, output_c, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        initialize_weights([self.conv5], 0.0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5


class INV_block(nn.Module):
    def __init__(self, subnet_constructor=ResidualDenseBlock_out, clamp=2.0, in_1=3, in_2=3):
        super(INV_block, self).__init__()
        self.split_len1 = in_1
        self.split_len2 = in_2
        self.clamp = clamp

        self.r = subnet_constructor(self.split_len1, self.split_len2)
        self.y = subnet_constructor(self.split_len1, self.split_len2)
        self.f = subnet_constructor(self.split_len2, self.split_len1)

    def e(self, s):
        return torch.exp(self.clamp * 2.0 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
        if not rev:
            t2 = self.f(x2)
            y1 = x1 + t2
            s1, t1 = self.r(y1), self.y(y1)
            y2 = self.e(s1) * x2 + t1
        else:
            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / self.e(s1)
            t2 = self.f(y2)
            y1 = x1 - t2
        return y1, y2
"""))

    # Cell 12: Markdown
    cells.append(nbf.v4.new_markdown_cell("""## 6. INet Full Model Architecture"""))

    # Cell 13: Code
    cells.append(nbf.v4.new_code_cell("""class INet(nn.Module):
    def __init__(self, in_c1=3, in_c2=3, block_num=[2, 1, 1]):
        super(INet, self).__init__()
        self.in_c1 = in_c1
        self.in_c2 = in_c2

        self.dwt = DWT()
        self.iwt = IWT()

        self.inv1_ops = nn.ModuleList([INV_block(in_1=in_c1, in_2=in_c2) for _ in range(block_num[0])])
        self.inv2_ops = nn.ModuleList([INV_block(in_1=in_c1, in_2=in_c2) for _ in range(block_num[1])])
        self.inv3_ops = nn.ModuleList([INV_block(in_1=in_c1, in_2=in_c2) for _ in range(block_num[2])])

        # Layer 3 frequency fusion
        self.cat_ll1 = nn.Conv2d(in_c1 * 2, in_c1, 1, 1, 0)
        self.cat_lh1 = nn.Conv2d(in_c1 * 2, in_c1, 1, 1, 0)
        self.cat_hl1 = nn.Conv2d(in_c1 * 2, in_c1, 1, 1, 0)
        self.cat_hh1 = nn.Conv2d(in_c1 * 2, in_c1, 1, 1, 0)

        # Layer 2 frequency fusion
        self.cat_ll2 = nn.Conv2d(in_c1 * 2, in_c1, 1, 1, 0)
        self.cat_lh2 = nn.Conv2d(in_c1 * 2, in_c1, 1, 1, 0)
        self.cat_hl2 = nn.Conv2d(in_c1 * 2, in_c1, 1, 1, 0)
        self.cat_hh2 = nn.Conv2d(in_c1 * 2, in_c1, 1, 1, 0)

        # Layer 1 frequency fusion
        self.cat_ll3 = nn.Conv2d(in_c1 * 2, in_c1, 1, 1, 0)
        self.cat_lh3 = nn.Conv2d(in_c1 * 2, in_c1, 1, 1, 0)
        self.cat_hl3 = nn.Conv2d(in_c1 * 2, in_c1, 1, 1, 0)
        self.cat_hh3 = nn.Conv2d(in_c1 * 2, in_c1, 1, 1, 0)

        # Progressive multi-scale reconstruction & upsampling
        self.up_conv1 = nn.Conv2d(in_c1, in_c1, 1, 1, 0)
        self.up_conv2 = nn.Conv2d(in_c1, in_c1, 1, 1, 0)
        self.up_conv3 = nn.Conv2d(in_c1, in_c1, 1, 1, 0)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.bn = nn.BatchNorm2d(in_c1)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.in_c1), x.narrow(1, self.in_c1, self.in_c2))

        if not rev:
            # Level 1 DWT (H/2, W/2)
            x1_LL1, x1_HL1, x1_LH1, x1_HH1 = self.dwt(x1)
            x2_LL1, x2_HL1, x2_LH1, x2_HH1 = self.dwt(x2)
            for op in self.inv1_ops:
                x1_LL1, x2_LL1 = op(torch.cat((x1_LL1, x2_LL1), dim=1))

            # Level 2 DWT (H/4, W/4)
            x1_LL2, x1_HL2, x1_LH2, x1_HH2 = self.dwt(x1_LL1)
            x2_LL2, x2_HL2, x2_LH2, x2_HH2 = self.dwt(x2_LL1)
            for op in self.inv2_ops:
                x1_LL2, x2_LL2 = op(torch.cat((x1_LL2, x2_LL2), dim=1))

            # Level 3 DWT (H/8, W/8)
            x1_LL3, x1_HL3, x1_LH3, x1_HH3 = self.dwt(x1_LL2)
            x2_LL3, x2_HL3, x2_LH3, x2_HH3 = self.dwt(x2_LL2)
            for op in self.inv3_ops:
                x1_LL3, x2_LL3 = op(torch.cat((x1_LL3, x2_LL3), dim=1))

            # Level 3 Fusion
            w_ll3 = torch.sigmoid(self.cat_ll1(torch.cat((x1_LL3, x2_LL3), dim=1)))
            x_LL3 = w_ll3 * x1_LL3 + (1.0 - w_ll3) * x2_LL3
            w_hl3 = torch.sigmoid(self.cat_hl1(torch.cat((x1_HL3, x2_HL3), dim=1)))
            x_HL3 = w_hl3 * x1_HL3 + (1.0 - w_hl3) * x2_HL3
            w_lh3 = torch.sigmoid(self.cat_lh1(torch.cat((x1_LH3, x2_LH3), dim=1)))
            x_LH3 = w_lh3 * x1_LH3 + (1.0 - w_lh3) * x2_LH3
            w_hh3 = torch.sigmoid(self.cat_hh1(torch.cat((x1_HH3, x2_HH3), dim=1)))
            x_HH3 = w_hh3 * x1_HH3 + (1.0 - w_hh3) * x2_HH3

            x_3 = self.iwt(torch.cat((x_LL3, x_HL3, x_LH3, x_HH3), dim=1))

            # Level 2 Fusion
            w_ll2 = torch.sigmoid(self.cat_ll2(torch.cat((x1_LL2, x2_LL2), dim=1)))
            x_LL2 = w_ll2 * x1_LL2 + (1.0 - w_ll2) * x2_LL2
            w_hl2 = torch.sigmoid(self.cat_hl2(torch.cat((x1_HL2, x2_HL2), dim=1)))
            x_HL2 = w_hl2 * x1_HL2 + (1.0 - w_hl2) * x2_HL2
            w_lh2 = torch.sigmoid(self.cat_lh2(torch.cat((x1_LH2, x2_LH2), dim=1)))
            x_LH2 = w_lh2 * x1_LH2 + (1.0 - w_lh2) * x2_LH2
            w_hh2 = torch.sigmoid(self.cat_hh2(torch.cat((x1_HH2, x2_HH2), dim=1)))
            x_HH2 = w_hh2 * x1_HH2 + (1.0 - w_hh2) * x2_HH2

            x_2 = self.iwt(torch.cat((x_LL2, x_HL2, x_LH2, x_HH2), dim=1))

            # Level 1 Fusion
            w_ll1 = torch.sigmoid(self.cat_ll3(torch.cat((x1_LL1, x2_LL1), dim=1)))
            x_LL1 = w_ll1 * x1_LL1 + (1.0 - w_ll1) * x2_LL1
            w_hl1 = torch.sigmoid(self.cat_hl3(torch.cat((x1_HL1, x2_HL1), dim=1)))
            x_HL1 = w_hl1 * x1_HL1 + (1.0 - w_hl1) * x2_HL1
            w_lh1 = torch.sigmoid(self.cat_lh3(torch.cat((x1_LH1, x2_LH1), dim=1)))
            x_LH1 = w_lh1 * x1_LH1 + (1.0 - w_lh1) * x2_LH1
            w_hh1 = torch.sigmoid(self.cat_hh3(torch.cat((x1_HH1, x2_HH1), dim=1)))
            x_HH1 = w_hh1 * x1_HH1 + (1.0 - w_hh1) * x2_HH1

            x_1 = self.iwt(torch.cat((x_LL1, x_HL1, x_LH1, x_HH1), dim=1))

            # Progressive multi-scale upsampling & skip reconstruction
            out = self.up_conv1(self.upsample(x_3)) + x_2
            out = self.up_conv2(self.upsample(out)) + x_1
            out = self.bn(out)
            return out


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = INet(in_c1=3, in_c2=3)

    def forward(self, x, rev=False):
        return self.model(x, rev=rev)


def init_model(mod, init_scale=0.01):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = init_scale * torch.randn_like(param.data)
            if len(split) >= 2 and split[-2] == 'conv5':
                param.data.fill_(0.0)

# Sanity check
test_net = Model().to(primary_device)
init_model(test_net)
sample_input = torch.randn(2, 6, 128, 128, device=primary_device)
with torch.no_grad():
    sample_out = test_net(sample_input)
total_params = sum(p.numel() for p in test_net.parameters() if p.requires_grad)
print(f"-> Model verified successfully! Input: {sample_input.shape} -> Output: {sample_out.shape}")
print(f"-> Total Trainable Parameters: {total_params:,}")
"""))

    # Cell 14: Markdown
    cells.append(nbf.v4.new_markdown_cell("""## 7. Loss Functions & Conditional Color Logic"""))

    # Cell 15: Code
    cells.append(nbf.v4.new_code_cell("""def rgb2ycbcr_t(img_rgb):
    R = img_rgb[:, 0, :, :].unsqueeze(1)
    G = img_rgb[:, 1, :, :].unsqueeze(1)
    B = img_rgb[:, 2, :, :].unsqueeze(1)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128.0 / 255.0
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128.0 / 255.0
    return Y, Cb, Cr


def rgb2ycbcr_np(img_rgb):
    # Expects [3, H, W] in [0, 255]
    R = img_rgb[0, :, :]
    G = img_rgb[1, :, :]
    B = img_rgb[2, :, :]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128.0 / 255.0
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128.0 / 255.0
    return Y, Cb, Cr


def guide_loss(output, target):
    return F.mse_loss(output, target)


def gradient(input_tensor):
    filter1 = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=input_tensor.device).view(1, 1, 3, 3)
    filter2 = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=input_tensor.device).view(1, 1, 3, 3)
    g1 = F.conv2d(input_tensor, filter1, bias=None, stride=1, padding=1)
    g2 = F.conv2d(input_tensor, filter2, bias=None, stride=1, padding=1)
    return torch.abs(g1) + torch.abs(g2)


def Grad_loss(output):
    out_grad = torch.mean(gradient(output), dim=[1, 2, 3])
    out_grad = 1.0 - torch.mean(out_grad / (out_grad + 1.0))
    return out_grad


def gaussian_1d(window_size, sigma):
    gauss = torch.tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel, device):
    _1D = gaussian_1d(window_size, 1.5).unsqueeze(1)
    _2D = _1D.mm(_1D.t()).float().unsqueeze(0).unsqueeze(0)
    return _2D.expand(channel, 1, window_size, window_size).contiguous().to(device)


def ssim_loss_calc(img1, img2, img3, window_size=11, channel=1):
    window = create_window(window_size, channel, img1.device)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu3 = F.conv2d(img3, window, padding=window_size // 2, groups=channel)

    mu1_sq, mu2_sq, mu3_sq = mu1.pow(2), mu2.pow(2), mu3.pow(2)
    mu1_mu2, mu1_mu3 = mu1 * mu2, mu1 * mu3

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma3_sq = F.conv2d(img3 * img3, window, padding=window_size // 2, groups=channel) - mu3_sq

    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    sigma13 = F.conv2d(img1 * img3, window, padding=window_size // 2, groups=channel) - mu1_mu3

    C2 = 0.03 ** 2
    x2 = torch.sqrt(torch.clamp(sigma2_sq, min=0))
    x3 = torch.sqrt(torch.clamp(sigma3_sq, min=0))

    ssim_map_12 = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map_13 = (2 * sigma13 + C2) / (sigma1_sq + sigma3_sq + C2)

    ssim_map_std = torch.where(torch.abs(x2) >= torch.abs(x3), ssim_map_12, ssim_map_13)
    return ssim_map_std.mean()
"""))

    # Cell 16: Markdown
    cells.append(nbf.v4.new_markdown_cell("""## 8. Validation Evaluation Metrics"""))

    # Cell 17: Code
    cells.append(nbf.v4.new_code_cell("""def calc_ssim_np(img1, img2, window_size=11, sigma=1.5):
    ax = np.arange(window_size) - window_size // 2
    kernel1d = np.exp(-ax ** 2 / (2.0 * sigma ** 2))
    kernel1d /= kernel1d.sum()
    kernel2d = np.outer(kernel1d, kernel1d)

    mu1 = correlate2d(img1, kernel2d, mode='valid')
    mu2 = correlate2d(img2, kernel2d, mode='valid')

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = correlate2d(img1 * img1, kernel2d, mode='valid') - mu1_sq
    sigma2_sq = correlate2d(img2 * img2, kernel2d, mode='valid') - mu2_sq
    sigma12 = correlate2d(img1 * img2, kernel2d, mode='valid') - mu1_mu2

    C1 = (0.01 * 255.0) ** 2
    C2 = (0.03 * 255.0) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-10)
    return float(np.mean(ssim_map))


def calc_viff_np(im1, im2, imf):
    EPS = 1e-10
    sq = 0.005 * 255.0 * 255.0
    p = np.array([1.0, 0.0, 0.15, 1.0]) / 2.15

    def get_lum(img):
        if img.ndim == 3:
            return 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
        return img.astype(np.float64)

    y1, y2, yf = get_lum(im1), get_lum(im2), get_lum(imf)

    def com_vid_vind(ref, dist):
        t_n, t_d, t_g = [], [], []
        sigma_nsq = sq
        for scale in range(1, 5):
            N = 2 ** (4 - scale + 1) + 1
            sig = N / 5.0
            k = np.arange(N) - N // 2
            g1d = np.exp(-k**2 / (2 * sig**2))
            g1d /= g1d.sum()
            win = np.outer(g1d, g1d)

            if scale > 1:
                ref = correlate2d(ref, win, mode='valid')[::2, ::2]
                dist = correlate2d(dist, win, mode='valid')[::2, ::2]

            mu1 = correlate2d(ref, win, mode='valid')
            mu2 = correlate2d(dist, win, mode='valid')
            s1_sq = np.maximum(correlate2d(ref * ref, win, mode='valid') - mu1 * mu1, 0)
            s2_sq = np.maximum(correlate2d(dist * dist, win, mode='valid') - mu2 * mu2, 0)
            s12 = correlate2d(ref * dist, win, mode='valid') - mu1 * mu2

            g = s12 / (s1_sq + EPS)
            sv_sq = s2_sq - g * s12
            g[s1_sq < EPS] = 0
            sv_sq[s1_sq < EPS] = s2_sq[s1_sq < EPS]
            s1_sq[s1_sq < EPS] = 0
            g[s2_sq < EPS] = 0
            sv_sq[s2_sq < EPS] = 0
            sv_sq[g < 0] = s2_sq[g < 0]
            g[g < 0] = 0
            sv_sq[sv_sq <= EPS] = EPS

            num = np.log10(1 + g**2 * s1_sq / (sv_sq + sigma_nsq))
            den = np.log10(1 + s1_sq / sigma_nsq)
            t_n.append(num)
            t_d.append(den)
            t_g.append(s1_sq)
        return t_n, t_d, t_g

    t1_n, t1_d, t1_g = com_vid_vind(y1, yf)
    t2_n, t2_d, t2_g = com_vid_vind(y2, yf)

    f_scales = np.zeros(4)
    for i in range(4):
        use_src1 = t1_g[i] < t2_g[i]
        m_z = np.where(use_src1, t1_n[i], t2_n[i])
        m_m = np.where(use_src1, t1_d[i], t2_d[i])
        vid = np.sum(m_z + 1e-7)
        vind = np.sum(m_m + 1e-7)
        f_scales[i] = vid / vind

    return float(np.dot(f_scales, p))


def calc_evaluator(A, B, F):
    A = np.array(A).astype(np.float32)
    B = np.array(B).astype(np.float32)
    F = np.array(F).astype(np.float32)

    # Spatial Frequency (SF)
    SF = np.sqrt(np.mean((F[:, 1:] - F[:, :-1]) ** 2) + np.mean((F[1:, :] - F[:-1, :]) ** 2))
    # Standard Deviation (SD)
    SD = np.std(F)

    # Average Gradient (AG)
    Gx = np.zeros_like(F)
    Gy = np.zeros_like(F)
    Gx[:, 0] = F[:, 1] - F[:, 0]
    Gx[:, -1] = F[:, -1] - F[:, -2]
    Gx[:, 1:-1] = (F[:, 2:] - F[:, :-2]) / 2.0
    Gy[0, :] = F[1, :] - F[0, :]
    Gy[-1, :] = F[-1, :] - F[-2, :]
    Gy[1:-1, :] = (F[2:, :] - F[:-2, :]) / 2.0
    AG_f = np.mean(np.sqrt((Gx ** 2 + Gy ** 2) / 2.0))

    # Cross Correlation (CC)
    rAF = np.sum((A - np.mean(A)) * (F - np.mean(F))) / (np.sqrt(np.sum((A - np.mean(A))**2) * np.sum((F - np.mean(F))**2)) + 1e-10)
    rBF = np.sum((B - np.mean(B)) * (F - np.mean(F))) / (np.sqrt(np.sum((B - np.mean(B))**2) * np.sum((F - np.mean(F))**2)) + 1e-10)
    CC_ALL = (rAF + rBF) / 2.0

    # Sum of Correlations of Differences (SCD)
    imgF_A = F - A
    imgF_B = F - B
    corr1 = np.sum((A - np.mean(A)) * (imgF_B - np.mean(imgF_B))) / (np.sqrt(np.sum((A - np.mean(A))**2) * np.sum((imgF_B - np.mean(imgF_B))**2)) + 1e-10)
    corr2 = np.sum((B - np.mean(B)) * (imgF_A - np.mean(imgF_A))) / (np.sqrt(np.sum((B - np.mean(B))**2) * np.sum((imgF_A - np.mean(imgF_A))**2)) + 1e-10)
    SCD_ALL = corr1 + corr2

    return float(SD + CC_ALL + SCD_ALL), float(SF + AG_f)
"""))

    # Cell 18: Markdown
    cells.append(nbf.v4.new_markdown_cell("""## 9. Distributed Multi-GPU Hyperparameters & Configuration"""))

    # Cell 19: Code
    cells.append(nbf.v4.new_code_cell(f"""# ==============================================================================
# Select Experiment Configuration
# Options: 'baseline' | 'job1_with_ct' | 'job2_with_ct'
# ==============================================================================
EXPERIMENT = '{experiment_name}'

CONFIG_PRESETS = {{
    'baseline': {{
        'tasks': ('PET-MRI', 'SPECT-MRI'),
        'return_color_flag': False,
        'alpha': 3.0,
        'beta': 1.0,
        'gamma': 1.0,
        'ks': 1.0,        # Standard color loss on PET/SPECT
        'mse_w': 0.01,
        'tag': 'baseline_paper_loss'
    }},
    'job1_with_ct': {{
        'tasks': ('PET-MRI', 'SPECT-MRI', 'CT-MRI'),
        'return_color_flag': False,
        'alpha': 3.0,
        'beta': 1.0,
        'gamma': 1.0,
        'ks': 0.0,        # NO color loss: 3 modalities (CT, PET, SPECT) trained with 0 color loss
        'mse_w': 0.01,
        'tag': 'job1_joint_ct_no_color'
    }},
    'job2_with_ct': {{
        'tasks': ('PET-MRI', 'SPECT-MRI', 'CT-MRI'),
        'return_color_flag': True,  # Enables conditional color loss
        'alpha': 3.0,
        'beta': 1.0,
        'gamma': 1.0,
        'ks': 1.0,        # Active on functional PET/SPECT, 0 on CT
        'mse_w': 0.01,
        'tag': 'job2_joint_ct_conditional_color'
    }}
}}

cfg = CONFIG_PRESETS[EXPERIMENT]

# Multi-GPU Distributed Setup
NUM_GPUS = torch.cuda.device_count()
USE_MULTI_GPU = (NUM_GPUS >= 2)
GPU_IDS = list(range(NUM_GPUS)) if NUM_GPUS > 0 else []

PER_GPU_BATCH = {per_gpu_batch}
BATCH_SIZE = PER_GPU_BATCH * max(1, NUM_GPUS)
NUM_WORKERS = min(8, 4 * max(1, NUM_GPUS))

# Scaled Hyperparameters
EPOCHS = 1000
CROPSIZE = 128
CROPSIZE_VAL = 256
BASE_LR = 10 ** -4.5       # 3.16227766e-5
LR = BASE_LR * math.sqrt(max(1, NUM_GPUS))  # Linear square-root scaling rule
WEIGHT_DECAY = 1e-5
BETAS = (0.9, 0.999)
WEIGHT_STEP = 1000
GAMMA_SCHED = 0.5
VAL_FREQ = {val_freq}

# Automatic Mixed Precision (AMP FP16) for Tensor Cores
USE_AMP = torch.cuda.is_available()

# Output directories
SAVE_ROOT = '/kaggle/working/checkpoints' if os.path.exists('/kaggle/working') else './checkpoints'
TIMESTAMP = time.strftime('%Y%m%d_%H%M%S')
EXP_SAVE_DIR = os.path.join(SAVE_ROOT, f"{{cfg['tag']}}_{{TIMESTAMP}}")
os.makedirs(EXP_SAVE_DIR, exist_ok=True)

print("=" * 70)
print(f"EXPERIMENT:      {{EXPERIMENT.upper()}} ({{cfg['tag']}})")
print(f"Tasks:           {{cfg['tasks']}}")
print(f"Multi-GPU Mode:  {{'ACTIVE (2x T4 GPUs)' if USE_MULTI_GPU else 'Single GPU / CPU'}}")
print(f"Active Devices:  {{GPU_IDS}}")
print(f"Batch Size:      {{BATCH_SIZE}} ({{PER_GPU_BATCH}} per GPU across {{max(1, NUM_GPUS)}} device(s))")
print(f"Data Workers:    {{NUM_WORKERS}}")
print(f"Learning Rate:   {{LR:.8f}} (Base: {{BASE_LR:.8f}}, Scaled by sqrt({{max(1, NUM_GPUS)}}))")
print(f"AMP FP16 Mode:   {{'ENABLED (Tensor Core Boost)' if USE_AMP else 'DISABLED'}}")
print(f"Loss Weights:    alpha={{cfg['alpha']}}, beta={{cfg['beta']}}, gamma={{cfg['gamma']}}, ks={{cfg['ks']}}, mse_w={{cfg['mse_w']}}")
print(f"Checkpoints Dir: {{EXP_SAVE_DIR}}")
print("=" * 70)
"""))

    # Cell 20: Markdown
    cells.append(nbf.v4.new_markdown_cell("""## 10. Execute Distributed INet Training Loop"""))

    # Cell 21: Code
    cells.append(nbf.v4.new_code_cell("""# 1. Prepare Distributed Data Loaders
trainloader, testloader = get_dataloaders(
    data_root=DATA_ROOT,
    tasks=cfg['tasks'],
    batch_size=BATCH_SIZE,
    cropsize=CROPSIZE,
    cropsize_val=CROPSIZE_VAL,
    return_color_flag=cfg['return_color_flag'],
    num_workers=NUM_WORKERS
)
print(f"Loaded {len(trainloader.dataset)} training pairs ({len(trainloader)} batches/epoch) and {len(testloader.dataset)} validation pairs.")

# 2. Instantiate Model and Wrap for Multi-GPU
net = Model().to(primary_device)
init_model(net)

if USE_MULTI_GPU:
    print(f"-> Wrapping Model in torch.nn.DataParallel across {NUM_GPUS} GPUs: {GPU_IDS}")
    net = nn.DataParallel(net, device_ids=GPU_IDS)

trainable_params = list(filter(lambda p: p.requires_grad, net.parameters()))
optimizer = optim.Adam(trainable_params, lr=LR, betas=BETAS, eps=1e-6, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=WEIGHT_STEP, gamma=GAMMA_SCHED)
scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

# 3. Training Loop Tracking
history = {
    'epoch': [], 'loss': [], 'ssim_loss': [], 'grad_loss': [], 'color_loss': [],
    'val_epoch': [], 'val_ssim': [], 'val_viff': [], 'val_sd_cc_scd': [], 'val_sf_ag': []
}

best_viff = -1.0
best_ssim = -1.0
best_model_path = os.path.join(EXP_SAVE_DIR, 'best_model.pt')
start_train_time = time.time()

print(f"\\nStarting distributed training loop on {max(1, NUM_GPUS)} GPU(s)...")
for epoch in range(1, EPOCHS + 1):
    net.train()
    loss_hist, r_hist, g_hist, l_hist, chro_hist = [], [], [], [], []

    for batch in trainloader:
        if cfg['return_color_flag']:
            data0, data1, is_col = batch
            is_col = is_col.to(primary_device)
        else:
            data0, data1 = batch
            is_col = None

        cover = data0.to(primary_device)
        secret = data1.to(primary_device)
        input_img = torch.cat((cover, secret), dim=1)

        optimizer.zero_grad()

        # Forward pass with AMP Mixed Precision
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            output = net(input_img)

            # Color Space Transformation
            F_Y, F_Cb, F_Cr = rgb2ycbcr_t(output)
            OTHER_Y, OTHER_Cb, OTHER_Cr = rgb2ycbcr_t(secret)
            MRI_Y = cover[:, 0:1, :, :]

            # SSIM Loss
            LOSS_SSIM = 1.0 - ssim_loss_calc(F_Y, MRI_Y, OTHER_Y)
            # Content Luminance Guide Loss
            mse_Y = guide_loss(F_Y, MRI_Y) + cfg['mse_w'] * guide_loss(F_Y, OTHER_Y)
            # Gradient Loss
            grad_loss = Grad_loss(F_Y)

            # Color Supervision:
            if cfg['ks'] == 0.0:
                # Job 1 with CT: NO color loss applied to any modality
                mse_chro = torch.tensor(0.0, device=primary_device)
            elif cfg['return_color_flag'] and is_col is not None:
                # Job 2 with CT: Conditional color loss (active on PET/SPECT, 0 on CT)
                diff_cb = (F_Cb - OTHER_Cb) ** 2
                diff_cr = (F_Cr - OTHER_Cr) ** 2
                diff_chro = (diff_cb + diff_cr).mean(dim=[1, 2, 3])
                col_count = is_col.sum()
                mse_chro = (diff_chro * is_col).sum() / (col_count + 1e-8) if col_count > 0 else torch.tensor(0.0, device=primary_device)
            else:
                # Baseline: Standard color loss across all pairs
                mse_chro = guide_loss(F_Cb, OTHER_Cb) + guide_loss(F_Cr, OTHER_Cr)

            l_loss = cfg['gamma'] * mse_Y + cfg['ks'] * mse_chro
            g_loss = grad_loss
            r_loss = LOSS_SSIM
            total_loss = cfg['alpha'] * r_loss + cfg['beta'] * g_loss + l_loss

        # Backward pass with Scaled Gradients
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_hist.append(total_loss.item())
        r_hist.append(r_loss.item())
        g_hist.append(g_loss.item())
        l_hist.append(l_loss.item())
        chro_hist.append(mse_chro.item())

    scheduler.step()

    avg_loss = np.mean(loss_hist)
    history['epoch'].append(epoch)
    history['loss'].append(avg_loss)
    history['ssim_loss'].append(np.mean(r_hist))
    history['grad_loss'].append(np.mean(g_hist))
    history['color_loss'].append(np.mean(chro_hist))

    # Log training status every epoch or every 10 epochs
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch [{epoch:4d}/{EPOCHS}] | Loss: {avg_loss:.4f} | SSIM_Loss: {np.mean(r_hist):.4f} | Grad_Loss: {np.mean(g_hist):.4f} | Color_Loss: {np.mean(chro_hist):.4f} | LR: {optimizer.param_groups[0]['lr']:.8f}")

    # Periodic Validation
    if epoch % VAL_FREQ == 0:
        net.eval()
        val_ssim_list, val_viff_list, metric_sd_cc_scd, metric_sf_ag = [], [], [], []

        with torch.no_grad():
            for val_batch in testloader:
                x0, x1 = val_batch[0].to(primary_device), val_batch[1].to(primary_device)
                val_input = torch.cat((x0, x1), dim=1)
                with torch.amp.autocast('cuda', enabled=USE_AMP):
                    val_out = net(val_input)

                # Convert to numpy uint8 range [0, 255]
                out_np = np.clip(val_out.cpu().float().numpy().squeeze() * 255.0, 0, 255)
                sec_np = np.clip(x1.cpu().float().numpy().squeeze() * 255.0, 0, 255)
                cov_np = np.clip(x0.cpu().float().numpy().squeeze() * 255.0, 0, 255)

                out_Y, _, _ = rgb2ycbcr_np(out_np)
                sec_Y, _, _ = rgb2ycbcr_np(sec_np)
                cov_Y, _, _ = rgb2ycbcr_np(cov_np)

                # Evaluation Metrics
                m_sd_cc_scd, m_sf_ag = calc_evaluator(sec_Y, cov_Y, out_Y)
                s_val = 0.5 * calc_ssim_np(cov_Y, out_Y) + 0.5 * calc_ssim_np(sec_Y, out_Y)
                v_val = calc_viff_np(cov_np, sec_np, out_np)

                val_ssim_list.append(s_val)
                val_viff_list.append(v_val)
                metric_sd_cc_scd.append(m_sd_cc_scd)
                metric_sf_ag.append(m_sf_ag)

        mean_ssim = np.mean(val_ssim_list)
        mean_viff = np.mean(val_viff_list)
        mean_s_c = np.mean(metric_sd_cc_scd)
        mean_s_a = np.mean(metric_sf_ag)

        history['val_epoch'].append(epoch)
        history['val_ssim'].append(mean_ssim)
        history['val_viff'].append(mean_viff)
        history['val_sd_cc_scd'].append(mean_s_c)
        history['val_sf_ag'].append(mean_s_a)

        is_best = ""
        raw_net = net.module if hasattr(net, 'module') else net
        if mean_viff > best_viff:
            best_viff = mean_viff
            best_ssim = mean_ssim
            is_best = " -> [BEST VIFF! Checkpoint Saved]"
            torch.save({
                'epoch': epoch,
                'model_state': raw_net.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'ssim': mean_ssim,
                'viff': mean_viff,
                'config': cfg
            }, best_model_path)

        print(f"  >>> [VAL @ Epoch {epoch:4d}] SSIM: {mean_ssim:.4f} | VIFF: {mean_viff:.4f} | SD+CC+SCD: {mean_s_c:.4f} | SF+AG: {mean_s_a:.4f}{is_best}")

    # Regular checkpoint every 100 epochs
    if epoch % 100 == 0:
        ckpt_path = os.path.join(EXP_SAVE_DIR, f'model_epoch_{epoch:04d}.pt')
        raw_net = net.module if hasattr(net, 'module') else net
        torch.save({'epoch': epoch, 'model_state': raw_net.state_dict(), 'optimizer_state': optimizer.state_dict()}, ckpt_path)

# Save final checkpoint
final_ckpt_path = os.path.join(EXP_SAVE_DIR, 'model_final.pt')
raw_net = net.module if hasattr(net, 'module') else net
torch.save({'epoch': EPOCHS, 'model_state': raw_net.state_dict(), 'optimizer_state': optimizer.state_dict(), 'config': cfg}, final_ckpt_path)

total_elapsed = (time.time() - start_train_time) / 60.0
print(f"\\n{'='*70}")
print(f"DISTRIBUTED TRAINING COMPLETED IN {total_elapsed:.2f} MINUTES!")
print(f"Best VIFF Score: {best_viff:.4f} (SSIM: {best_ssim:.4f})")
print(f"Checkpoints directory: {EXP_SAVE_DIR}")
print(f"{'='*70}")
"""))

    # Cell 22: Markdown
    cells.append(nbf.v4.new_markdown_cell("""## 11. Convergence & Training Metrics Visualization"""))

    # Cell 23: Code
    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Total Loss Curve
axes[0].plot(history['epoch'], history['loss'], color='#1f77b4', lw=2, label='Total Loss')
axes[0].set_title('Training Loss Convergence', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].legend()

# Loss Components
axes[1].plot(history['epoch'], history['ssim_loss'], color='#2ca02c', label='SSIM Loss')
axes[1].plot(history['epoch'], history['grad_loss'], color='#d62728', label='Gradient Loss')
axes[1].plot(history['epoch'], history['color_loss'], color='#9467bd', label='Color Loss')
axes[1].set_title('Loss Component Breakdown', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss Value')
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].legend()

# Validation Scores
if history['val_epoch']:
    axes[2].plot(history['val_epoch'], history['val_viff'], marker='o', color='#ff7f0e', lw=2, label='VIFF')
    axes[2].plot(history['val_epoch'], history['val_ssim'], marker='s', color='#17becf', lw=2, label='SSIM')
    axes[2].set_title('Validation Performance', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Metric Score')
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].legend()

plt.tight_layout()
plot_path = os.path.join(EXP_SAVE_DIR, 'training_curves.png')
plt.savefig(plot_path, dpi=300)
plt.show()
print(f"Saved training plot to: {plot_path}")
"""))

    # Cell 24: Markdown
    cells.append(nbf.v4.new_markdown_cell("""## 12. Visual Evaluation & Inspection of Fusion Results"""))

    # Cell 25: Code
    cells.append(nbf.v4.new_code_cell("""# Load the Best Checkpoint for Evaluation
eval_net = Model().to(primary_device)
ckpt_to_load = best_model_path if os.path.exists(best_model_path) else final_ckpt_path
state = torch.load(ckpt_to_load, map_location=primary_device)
model_weights = state.get('model_state', state.get('net', state))
clean_weights = {k.replace('module.', ''): v for k, v in model_weights.items() if 'tmp_var' not in k}
eval_net.load_state_dict(clean_weights)
eval_net.eval()
print(f"Loaded evaluation checkpoint from: {ckpt_to_load}")

# Test visualizer for multiple pairs
sample_indices = min(5, len(testloader))
fig, axes = plt.subplots(sample_indices, 3, figsize=(12, 4 * sample_indices))

with torch.no_grad():
    for idx, batch in enumerate(testloader):
        if idx >= sample_indices:
            break
        cov_t, sec_t = batch[0].to(primary_device), batch[1].to(primary_device)
        inp = torch.cat((cov_t, sec_t), dim=1)
        fused_t = eval_net(inp)

        cov_img = cov_t.cpu().squeeze().permute(1, 2, 0).numpy()
        sec_img = sec_t.cpu().squeeze().permute(1, 2, 0).numpy()
        fused_img = np.clip(fused_t.cpu().squeeze().permute(1, 2, 0).numpy(), 0.0, 1.0)

        # Plot Cover (MRI)
        ax_mri = axes[idx, 0] if sample_indices > 1 else axes[0]
        ax_mri.imshow(cov_img)
        ax_mri.set_title(f"Pair #{idx+1}: MRI (Cover)", fontsize=11, fontweight='bold')
        ax_mri.axis('off')

        # Plot Secret (Functional / CT)
        ax_sec = axes[idx, 1] if sample_indices > 1 else axes[1]
        ax_sec.imshow(sec_img)
        ax_sec.set_title(f"Pair #{idx+1}: Secret Modality", fontsize=11, fontweight='bold')
        ax_sec.axis('off')

        # Plot Fused Output
        ax_fus = axes[idx, 2] if sample_indices > 1 else axes[2]
        ax_fus.imshow(fused_img)
        ax_fus.set_title(f"Pair #{idx+1}: INet Fusion Output", fontsize=11, fontweight='bold')
        ax_fus.axis('off')

plt.tight_layout()
results_plot_path = os.path.join(EXP_SAVE_DIR, 'visual_fusion_results.png')
plt.savefig(results_plot_path, dpi=300)
plt.show()
print(f"Saved visual fusion samples to: {results_plot_path}")
"""))

    nb['cells'] = cells
    nb['metadata'] = {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.11.0'
        },
        'accelerator': 'GPU'
    }
    return nb


def main():
    # 1. Generate master notebook
    master_nb = build_notebook(experiment_name='job1_with_ct', per_gpu_batch=128, val_freq=100)
    os.makedirs('notebooks', exist_ok=True)
    with open('notebooks/train_inet_kaggle.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(master_nb, f)
    print("Generated master notebook: notebooks/train_inet_kaggle.ipynb")

    # 2. Setup 3 independent Kaggle jobs
    jobs = [
        {
            'dir': 'kaggle_jobs/job1_baseline',
            'slug': 'mmif-inet-task-1-baseline',
            'title': 'MMIF-INet Task 1 Baseline',
            'exp': 'baseline',
            'file': 'train_inet_baseline.ipynb'
        },
        {
            'dir': 'kaggle_jobs/job2_ct_no_color',
            'slug': 'mmif-inet-task-2-ct-no-color',
            'title': 'MMIF-INet Task 2 CT No Color',
            'exp': 'job1_with_ct',
            'file': 'train_inet_job1_with_ct.ipynb'
        },
        {
            'dir': 'kaggle_jobs/job3_ct_conditional_color',
            'slug': 'mmif-inet-task-3-ct-conditional-color',
            'title': 'MMIF-INet Task 3 CT Conditional Color',
            'exp': 'job2_with_ct',
            'file': 'train_inet_job2_with_ct.ipynb'
        }
    ]

    for job in jobs:
        os.makedirs(job['dir'], exist_ok=True)
        # Build specific notebook
        job_nb = build_notebook(experiment_name=job['exp'], per_gpu_batch=128, val_freq=100)
        nb_path = os.path.join(job['dir'], job['file'])
        with open(nb_path, 'w', encoding='utf-8') as f:
            nbf.write(job_nb, f)

        # Write kernel metadata
        meta = {
            "id": f"cyclerlol/{job['slug']}",
            "title": job['title'],
            "code_file": job['file'],
            "language": "python",
            "kernel_type": "notebook",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_tpu": "false",
            "enable_internet": "true",
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": []
        }
        meta_path = os.path.join(job['dir'], 'kernel-metadata.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)

        print(f"Set up Kaggle Job [{job['slug']}] in {job['dir']}")


if __name__ == '__main__':
    main()

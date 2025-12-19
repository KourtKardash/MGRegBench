import os, glob
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "0"

import torch, sys
from torch.utils.data import Dataset

import numpy as np
import ants

import random

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)


class MammoDataset(Dataset):
    def __init__(self, root_dir, transforms = None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.subfolders = []
        for subdir in ['INBreast', 'KAU-BCMD', 'RSNA']:
            subdir_path = os.path.join(root_dir, subdir)
            pair_dirs = glob.glob(os.path.join(subdir_path, '*'))
            self.subfolders.extend(pair_dirs)

    def __getitem__(self, index):
        subfolder = random.choice(self.subfolders)
        images = glob.glob(os.path.join(subfolder, '*.jpg')) + glob.glob(os.path.join(subfolder, '*.png'))
        indices = np.random.choice(len(images), size=2, replace=False)

        fixed_img_path = images[indices[0]]
        moving_img_path = images[indices[1]]

        fixed_img = self.load_image(fixed_img_path)
        moving_img = self.load_image(moving_img_path)

        fixed_ants = ants.from_numpy(fixed_img)
        moving_ants = ants.from_numpy(moving_img)

        reg = ants.registration(
            fixed=fixed_ants,
            moving=moving_ants,
            type_of_transform="Affine",
            random_seed=42
        )

        moving_ants_warped = reg['warpedmovout']
        moving_img = moving_ants_warped.numpy()

        img_fixed = torch.from_numpy(fixed_img).unsqueeze(0)
        img_fixed = img_fixed.to(torch.float32)
        img_moving = torch.from_numpy(moving_img).unsqueeze(0)
        img_moving = img_moving.to(torch.float32)

        img_mov_id = str(indices[1])
        img_fix_id = str(indices[0])

        data = {
            "img_fix": img_fixed,
            "img_mov": img_moving,
            "img_fix_id": img_fix_id,
            "img_mov_id": img_mov_id,
        }
        return data

    def load_image(self, path):
        from PIL import Image        
        img = Image.open(path).convert('L')
        img = img.resize((512, 1024), Image.Resampling.LANCZOS)
        img = np.array(img, dtype=np.float32)
        img = img/255.0
        return img

    def __len__(self):
        return 1500
import os
import json

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
#os.environ["PYTHONHASHSEED"] = "0"
#os.environ['TMPDIR'] = '/home/s.krasnova/tmp'

import torch
import numpy as np
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

import xml.etree.ElementTree as ET
import pandas as pd

from scipy.interpolate import RegularGridInterpolator
import shutil

import itertools
from PIL import Image

from pathlib import Path
import time

from skimage.metrics import normalized_mutual_information as mi
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

from Methods.MammoRegNet.utils.utils import *
from Methods.MammoRegNet.train.losses_mammoregnet import  *
from Methods.MammoRegNet.models.MammoRegNet import MammoRegNet

from .rTRE import calc_tre


def analyze_deformation_field_2d(warp):
    u = warp[..., 0]  # displacement in Y (rows)
    v = warp[..., 1]  # displacement in X (columns)

    du_dy, du_dx = np.gradient(u)
    dv_dy, dv_dx = np.gradient(v)

    jacobian_det = (1 + du_dy) * (1 + dv_dx) - du_dx * dv_dy

    percent_folded = 100.0 * np.sum(jacobian_det < 0) / jacobian_det.size
    return float(percent_folded)


def dice_coefficient(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2).sum()
    size1 = mask1.sum()
    size2 = mask2.sum()
    dice = (2.0 * intersection) / (size1 + size2)
    return dice


def merge_two_filenames(f1, f2):
    f1 = f1.strip()
    f2 = f2.strip()

    parts1 = f1.split('_', 2)
    parts2 = f2.split('_', 2)

    year1 = parts1[0]
    year2 = parts2[0]

    rest = '_'.join(parts1[1:])
    return f"{year1}_{year2}_{rest}"


def apply_transform_to_landmarks(warp, image_name, output_xml_path):
    tree = ET.parse(output_xml_path)
    root = tree.getroot()

    for image in root.findall('image'):
        if image.get('name') == image_name:
            points_tags = image.findall('points')
            point_list = []
            index_to_tag = []
            for points_tag in points_tags:
                x, y = map(float, points_tag.get('points').split(','))
                point_list.append([int(round(y)), int(round(x))])
                index_to_tag.append(points_tag)
                
            points_df = pd.DataFrame(point_list, columns=['x', 'y'])
            points = points_df.values 

            width = int(image.get('width'))
            height = int(image.get('height'))
            k1 = 512.0/width
            k2 = 1024.0/height
            image.set('width', str(512))
            image.set('height', str(1024))

            points[:, 0] = points[:, 0] * k2
            points[:, 1] = points[:, 1] * k1

            height, width, _ = warp.shape
            y_coords, x_coords = np.arange(height), np.arange(width)
            interp_dx = RegularGridInterpolator((y_coords, x_coords), warp[..., 1],\
                                                 method='linear', bounds_error=False, fill_value=0)
            interp_dy = RegularGridInterpolator((y_coords, x_coords), warp[..., 0],\
                                                 method='linear', bounds_error=False, fill_value=0)

            dx = interp_dx(points)
            dy = interp_dy(points)

            deformed_points = points + np.column_stack((dy, dx))

            for i, points_tag in enumerate(index_to_tag):
                new_x = deformed_points[i][0]
                new_y = deformed_points[i][1]
                points_tag.set('points', f"{new_y:.2f},{new_x:.2f}")

    tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)


def registration(before_img, after_img, moving_mask, fixed_mask, output_path,
                 f1_path, f2_path, p_mask, best_model, output_xml1, output_xml2):
    t = time.time()
    image_name = f1_path.parent.name
    fixed_img = after_img
    moving_img = before_img
    
    img1 = Image.fromarray(before_img)
    img2 = Image.fromarray(after_img)

    img1.save(f1_path)
    img2.save(f2_path)
    
    before_img = Image.fromarray(before_img)
    after_img = Image.fromarray(after_img)
  
    moving_mask = Image.fromarray(moving_mask)
    fixed_mask = Image.fromarray(fixed_mask)

    before_img = before_img.resize((512, 1024), Image.Resampling.LANCZOS)
    after_img = after_img.resize((512, 1024), Image.Resampling.LANCZOS)

    moving_mask = moving_mask.resize((512, 1024), Image.Resampling.NEAREST)
    fixed_mask = fixed_mask.resize((512, 1024), Image.Resampling.NEAREST)

    after_img.save(f2_path)

    before_img = np.array(before_img)
    after_img = np.array(after_img)
    moving_mask = np.array(moving_mask)
    fixed_mask = np.array(fixed_mask)

    before_img = before_img/255.0
    after_img = after_img/255.0

    before_img = np.expand_dims(before_img, axis=-1)
    after_img = np.expand_dims(after_img, axis=-1)
    moving_mask = np.expand_dims(moving_mask, axis=-1)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    device = 'cuda'
    
    model = MammoRegNet()
    model.load_state_dict(torch.load(best_model, map_location=torch.device(device)))
    model = model.to(device)
    model.eval()

    before_img = before_img[np.newaxis, ...]
    after_img = after_img[np.newaxis, ...]
    moving_mask = moving_mask[np.newaxis, ...]

    # set up tensors and permute
    input_moving = torch.from_numpy(before_img).to(device).float().permute(0, 3, 1, 2)
    input_fixed = torch.from_numpy(after_img).to(device).float().permute(0, 3, 1, 2)
    tensor_mask = torch.from_numpy(moving_mask).to(device).float().permute(0, 3, 1, 2)

    with torch.no_grad():
        # predict
        pred = model(input_fixed, input_moving, tensor_mask)
        moved = pred[0][0]
        mask_warped = pred[4][0]
        warp = pred[1][0]

        moved = moved.detach().cpu().numpy().squeeze().squeeze()
        mask_warped = mask_warped.detach().cpu().numpy().squeeze().squeeze()
        warp = warp.unsqueeze(0).permute(0, 2, 3, 1).cpu().numpy()

    runtime = time.time() - t

    if torch.cuda.is_available():
        peak_memory_allocated = torch.cuda.max_memory_allocated("cuda") 
        peak_memory_allocated = peak_memory_allocated / 1024 / 1024 / 1024 #Gb
    else:
        peak_memory_allocated = 0
    
    im = Image.fromarray((moved * 255).astype(np.uint8))
    im.save(output_path)    
    after_img = after_img.squeeze().squeeze()

    image = Image.fromarray((mask_warped * 255).astype(np.uint8))
    image.save(p_mask)

    after_img = (after_img * 255).astype(np.uint8)
    moved = (moved * 255).astype(np.uint8)
    per = analyze_deformation_field_2d(warp.squeeze(0))
    apply_transform_to_landmarks(warp.squeeze(0), image_name + '.png', output_xml1)
    apply_transform_to_landmarks(warp.squeeze(0), image_name + '.png', output_xml2)

    return mse(after_img, moved), ssim(after_img, moved, data_range=255),\
           mi(after_img, moved), np.corrcoef(after_img.flatten(), moved.flatten())[0, 1],\
           dice_coefficient(mask_warped, fixed_mask), per, peak_memory_allocated, runtime


def process_annotations(xml_path, moving=False):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for image in root.findall('image'):
        width = int(image.get('width'))
        height = int(image.get('height'))
        k1 = 512.0/width
        k2 = 1024.0/height

        for points in image.findall('points'):
            points_str = points.get('points')
            x, y = points_str.split(',')
            x = float(x)
            y = float(y)
            if not moving and width > 1000:
                x = x / 2
                y = y / 2
            if moving:
                x = x * k1
                y = y * k2

            points.set('points', f"{x:.2f},{y:.2f}")

        if not moving and width > 1000:
            image.set('width', str(width // 2))
            image.set('height', str(height // 2))
        if moving:
            image.set('width', str(512))
            image.set('height', str(1024))

    tree.write(xml_path, encoding='utf-8', xml_declaration=True)


def main():
    output_folder = "outputs/MammoRegNet/"
    os.makedirs(output_folder, exist_ok=True)
    best_model = output_folder + 'models/' + 'model_1499_original.pth'
    base_dir = Path("Dataset/MGRegBench/evaluation")
    masks_dir = Path("Dataset/MGRegBench/evaluation-masks")
    subdatasets = ["INBreast", "KAU-BCMD", "RSNA"]
    json_file = output_folder + 'mammoregnet.json'

    data_path = "Dataset/MGRegBench/"
    shutil.copy2(data_path + 'moving_landmarks.xml', output_folder + 'moving_landmarks.xml')
    shutil.copy2(data_path + 'fixed_landmarks_1.xml', output_folder + 'fixed_landmarks_1.xml')
    shutil.copy2(data_path + 'fixed_landmarks_2.xml', output_folder + 'fixed_landmarks_2.xml')
    process_annotations(output_folder + 'moving_landmarks.xml', True)
    process_annotations(output_folder + 'fixed_landmarks_1.xml')
    process_annotations(output_folder + 'fixed_landmarks_2.xml')

    mse_list = []
    ssim_list = []
    mi_list = []
    cc_list = []
    dice_list = []
    njd_list = []
    vram_list = []
    runtime_list = []

    for subdataset in subdatasets:
        subdataset_path = base_dir / subdataset
        if not subdataset_path.exists():
            print(f"Folder not found: {subdataset_path}")
            continue

        print(f"\n{'='*50}")
        print(f"PROCESSING: {subdataset}")
        print(f"{'='*50}")

        for patient_folder in sorted(subdataset_path.iterdir()):
            if not patient_folder.is_dir():
                continue

            patient_id = patient_folder.name
            image_files = []
            for ext in ["*.jpg", "*.png"]:
                image_files.extend(patient_folder.glob(ext))
            
            file_names = sorted([f for f in image_files])
            f1 = file_names[0]
            f2 = file_names[1]

            print(f"\n[Patient: {patient_id}]")

            before_img = Image.open(f1).convert('L')
            after_img = Image.open(f2).convert('L')

            #before_img = before_img.resize((512, 1024), Image.Resampling.LANCZOS)
            #after_img = after_img.resize((512, 1024), Image.Resampling.LANCZOS)

            min_height = min(before_img.height, after_img.height)
            min_width = min(before_img.width, after_img.width)
               
            if min_height > 1000:
                before_img = before_img.resize((min_width // 2, min_height // 2), Image.Resampling.LANCZOS)
                after_img = after_img.resize((min_width // 2, min_height // 2), Image.Resampling.LANCZOS)
    
            before_arr = np.array(before_img)
            after_arr = np.array(after_img)

            output_dir = Path(output_folder) / subdataset / patient_id
            output_dir.mkdir(parents=True, exist_ok=True)
            f1_path = output_dir / f"{f1.stem}.png"
            f2_path = output_dir / f"{f2.stem}.png"

            m1 = Image.open(masks_dir / subdataset / patient_id / f"{f1.stem}.png").convert('L')
            m2 = Image.open(masks_dir / subdataset / patient_id / f"{f2.stem}.png").convert('L')

            #m1 = m1.resize((512, 1024), Image.Resampling.NEAREST)
            #m2 = m2.resize((512, 1024), Image.Resampling.NEAREST)
            if min_height > 1000:
                m1 = m1.resize((min_width // 2, min_height // 2), Image.Resampling.LANCZOS)
                m2 = m2.resize((min_width // 2, min_height // 2), Image.Resampling.LANCZOS)

            m1_arr = np.array(m1)
            m2_arr = np.array(m2)
            m1_arr = (m1_arr > 180).astype(np.float32)
            m2_arr = (m2_arr > 180).astype(np.float32)

            p = output_dir / "mask.png"

            if subdataset == "RSNA":
                output_path = output_dir / f"{f1.stem}_{f2.stem}.png"
                before_arr = before_arr * (m1_arr).astype(np.uint8)
                after_arr = after_arr * (m2_arr).astype(np.uint8)
            else:
                name = merge_two_filenames(f1.stem, f2.stem)
                output_path = output_dir / f"{name}.png"

            try:
                mse, ssim, mi, cc, dice, njd, vram, runtime =\
                registration(before_arr, after_arr, m1_arr, m2_arr,
                             output_path, f1_path, f2_path, p, best_model,
                             output_folder + 'fixed_landmarks_1.xml',
                             output_folder + 'fixed_landmarks_2.xml')

                if os.path.exists(json_file):
                    with open(json_file, 'r', encoding='utf-8') as f:
                        try:
                            results = json.load(f)
                        except json.JSONDecodeError:
                            results = {}
                else:
                    results = {}

                results[str(output_path)] = {
                    "MSE": round(mse, 6),
                    "SSIM": round(ssim, 6),
                    "MI": round(mi, 6),
                    "CC": round(cc, 6),
                    "DSC": round(dice, 6),
                    "NJD": round(njd, 6),
                    "VRAM": round(vram, 6),
                    "Runtime": round(runtime, 6)
                }
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

                mse_list.append(mse)
                ssim_list.append(ssim)
                mi_list.append(mi)
                cc_list.append(cc)
                dice_list.append(dice)
                njd_list.append(njd)
                vram_list.append(vram)
                runtime_list.append(runtime)

            except Exception as e:
                print(f"Error creating images for {patient_id}: {e}")

    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = {}
    else:
        results = {}

    results["Mean Metrics"] = {
                            "MSE": round(np.mean(mse_list), 6),
                            "SSIM": round(np.mean(ssim_list), 6),
                            "MI": round(np.mean(mi_list), 6),
                            "CC": round(np.mean(cc_list), 6),
                            "DSC": round(np.mean(dice_list), 6),
                            "NJD": round(np.mean(njd_list), 6),
                            "VRAM": round(np.mean(vram_list), 6),
                            "Runtime": round(np.mean(runtime_list), 6)
                        }

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Mean MSE: {np.mean(mse_list):.4f}")
    print(f"Mean SSIM: {np.mean(ssim_list):.4f}")
    print(f"Mean Mutual Information: {np.mean(mi_list):.4f}")
    print(f"Mean Correlation Coefficient: {np.mean(cc_list):.4f}")
    print(f"Mean dice: {np.mean(dice_list):.4f}")
    print(f"Mean njd: {np.mean(njd_list):.4f}")
    print(f"Mean vram: {np.mean(vram_list):.4f}")
    print(f"Mean runtime: {np.mean(runtime_list):.4f}")

    calc_tre(output_folder + 'moving_landmarks.xml',
             output_folder + 'fixed_landmarks_1.xml',
             output_folder + 'fixed_landmarks_2.xml')



if __name__ == "__main__":
    main()

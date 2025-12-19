import os

#os.environ['TMPDIR'] = '/home/s.krasnova/tmp'

import ants
import argparse
import json
from models import models
import numpy as np
import pandas as pd
from pathlib import Path
import torch

import shutil
from PIL import Image
import time
import xml.etree.ElementTree as ET

from skimage.metrics import normalized_mutual_information as mi
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse


def compute_landmarks(network, landmarks_pre, image_size):
    scale_of_axes = [(0.5 * s) for s in image_size]
    coordinate_tensor = torch.FloatTensor(landmarks_pre / (scale_of_axes)) - 1.0
    output = network(coordinate_tensor.cuda())
    delta = output.cpu().detach().numpy() * (scale_of_axes)
    return landmarks_pre + delta, delta


def analyze_deformation_field_2d(deformation_data, warped):
    H , W = warped.shape
    deformation_data = deformation_data.reshape(H, W, 2)
    u_y = deformation_data[..., 0]
    u_x = deformation_data[..., 1]

    duy_dy, duy_dx = np.gradient(u_y)
    dux_dy, dux_dx = np.gradient(u_x)

    jacobian_det = (1 + dux_dx) * (1 + duy_dy) - dux_dy * duy_dx

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


def apply_affine_transform_to_landmarks(affine_path, image_name,
                                        output_xml_path):
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
            transformed_points = ants.apply_transforms_to_points(
                dim=2,
                points=points_df,
                transformlist=[affine_path]
            ) 
            for i, points_tag in enumerate(index_to_tag):
                new_x = transformed_points['x'].iloc[i]
                new_y = transformed_points['y'].iloc[i]
                points_tag.set('points', f"{new_y:.2f},{new_x:.2f}")

    tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)


def apply_warp_transform_to_landmarks(reg, image_name, shape, output_xml_path):
    tree = ET.parse(output_xml_path)
    root = tree.getroot()

    for image in root.findall('image'):
        if image.get('name') == image_name:
            w = int(image.get('width'))
            points_tags = image.findall('points')
            point_list = []
            index_to_tag = []
            for points_tag in points_tags:
                x, y = map(float, points_tag.get('points').split(','))
                point_list.append([int(round(y)), int(round(x))])
                index_to_tag.append(points_tag)
                
            points_array = np.array(point_list)
            transformed_points, _ = compute_landmarks(
                reg, points_array, image_size=shape
            )
            for i, points_tag in enumerate(index_to_tag):
                new_x = transformed_points[i][0]
                new_y = transformed_points[i][1]
                points_tag.set('points', f"{new_y:.2f},{new_x:.2f}")

    tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)


def load_mammogram_pair(dataset, old_path, new_path, moving_mask_path, fixed_mask_path, xml_path1, xml_path2, affine_path):
    img_old = Image.open(old_path).convert('L')
    img_new = Image.open(new_path).convert('L')
    width, height = img_old.size

    min_height = min(img_old.height, img_new.height)
    min_width = min(img_old.width, img_new.width)
                
    fixed_mask = Image.open(fixed_mask_path).convert('L')
    moving_mask = Image.open(moving_mask_path).convert('L')

    if min_height > 1000:
        img_old = img_old.resize((min_width // 2, min_height // 2),
                                       Image.Resampling.LANCZOS)
        img_new = img_new.resize((min_width // 2, min_height // 2),
                                     Image.Resampling.LANCZOS)
        fixed_mask = fixed_mask.resize((min_width // 2, min_height // 2),
                                       Image.Resampling.LANCZOS)
        moving_mask = moving_mask.resize((min_width // 2, min_height // 2),
                                         Image.Resampling.LANCZOS)
    
    img_old = np.array(img_old)
    img_new = np.array(img_new)
    fixed_mask = np.array(fixed_mask)
    moving_mask = np.array(moving_mask)
    fixed_mask = (fixed_mask > 180).astype(np.uint8)
    moving_mask = (moving_mask > 180).astype(np.uint8)

    if dataset == 'RSNA':
        img_old = img_old * moving_mask
        img_new = img_new * fixed_mask

    fixed_ants = ants.from_numpy(img_new)
    moving_ants = ants.from_numpy(img_old)

    warped_img = ants.apply_transforms(
        fixed=fixed_ants,
        moving=moving_ants,
        transformlist=[affine_path],
        interpolator='linear'
    )

    warped_np = warped_img.numpy()
    warped = warped_np.astype(np.uint8)
    
    print(mse(img_new, warped))

    image_name = old_path.parent.name
    apply_affine_transform_to_landmarks(affine_path, image_name + '.png', xml_path1)
    apply_affine_transform_to_landmarks(affine_path, image_name + '.png', xml_path2)

    img_old = warped

    img_old = img_old / 255.0
    img_new = img_new / 255.0

    mask_ants = ants.from_numpy(moving_mask)
    warped_mask = ants.apply_transforms(
        fixed=fixed_ants,
        moving=mask_ants,
        transformlist=[affine_path],
        interpolator='genericLabel',
        verbose=False
    )

    moving_mask = warped_mask.numpy()
    moving_mask = moving_mask.astype(np.float32)
    fixed_mask = fixed_mask.astype(np.float32)
    return img_new, img_old, moving_mask, fixed_mask

def process_annotations(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for image in root.findall('image'):
        width = int(image.get('width'))
        height = int(image.get('height'))
        for points in image.findall('points'):
            points_str = points.get('points')
            x, y = points_str.split(',')
            x = float(x)
            y = float(y)
            if width > 1000:
                x = x / 2
                y = y / 2
            points.set('points', f"{x:.2f},{y:.2f}")
        if width > 1000:
            image.set('width', str(width // 2))
            image.set('height', str(height // 2))

    tree.write(xml_path, encoding='utf-8', xml_declaration=True)


parser = argparse.ArgumentParser(description="Choosing launch index")
parser.add_argument("--index", type=int, default=11)
args = parser.parse_args()

data_path = "Dataset/MGRegBench/"
output_folder = f"outputs/IDIR/results_{args.index}/"
os.makedirs(output_folder, exist_ok=True)
shutil.copy2(data_path + 'moving_landmarks.xml', output_folder + 'moving_landmarks.xml')
shutil.copy2(data_path + 'fixed_landmarks_1.xml', output_folder + 'fixed_landmarks_1.xml')
shutil.copy2(data_path + 'fixed_landmarks_2.xml', output_folder + 'fixed_landmarks_2.xml')
process_annotations(output_folder + 'moving_landmarks.xml')
process_annotations(output_folder + 'fixed_landmarks_1.xml')
process_annotations(output_folder + 'fixed_landmarks_2.xml')

base_dir = Path("Dataset/MGRegBench/evaluation")
masks_dir = Path("Dataset/MGRegBench/evaluation-masks")
subdatasets = ["INBreast", "KAU-BCMD", "RSNA"]
json_file = output_folder + 'IDIR.json'
mse_list = []
ssim_list = []
mi_list = []
cc_list = []
dice_list = []
njd_list = []
vram_list = []
runtime_list = []
results = {}

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
        output_dir = Path(output_folder) / subdataset / patient_id
        output_dir.mkdir(parents=True, exist_ok=True)

        image_files = []
        for ext in ["*.jpg", "*.png"]:
            image_files.extend(patient_folder.glob(ext))
        file_names = sorted([f for f in image_files])
        f1 = file_names[0]
        f2 = file_names[1]
        affine_path = subdataset_path / patient_id / "affine.mat"

        m1 = masks_dir / subdataset / patient_id / f"{f1.stem}.png"
        m2 = masks_dir / subdataset / patient_id / f"{f2.stem}.png"
        img_insp, img_exp, mask_exp, mask_insp = load_mammogram_pair(subdataset, f1, f2, m1, m2,
                                                                     output_folder + 'fixed_landmarks_1.xml',
                                                                     output_folder + 'fixed_landmarks_2.xml', str(affine_path))

        def merge_two_filenames(f1, f2):
            f1 = f1.strip()
            f2 = f2.strip()
            parts1 = f1.split('_', 2)
            parts2 = f2.split('_', 2)
            year1 = parts1[0]
            year2 = parts2[0]
            rest = '_'.join(parts1[1:])
            return f"{year1}_{year2}_{rest}"
        
        if subdataset == "RSNA":
            output_path = output_dir / f"{f1.stem}_{f2.stem}.png"
            output_path_mask = output_dir / "mask.png"
        else:
            name = merge_two_filenames(f1.stem, f2.stem)
            output_path = output_dir / f"{name}.png"
            output_path_mask = output_dir / "mask.png"

        img_insp_tensor = torch.from_numpy(img_insp).float()
        img_exp_tensor = torch.from_numpy(img_exp).float()
        kwargs = {
            "verbose": False,
            "hyper_regularization": False,
            "jacobian_regularization": True,
            "bending_regularization": False,
            "network_type": "MLP",
            "save_folder": output_folder,
            "mask": mask_exp,
            "image_shape": img_insp.shape
        }

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        t = time.time()

        ImpReg = models.ImplicitRegistrator(img_exp_tensor, img_insp_tensor, **kwargs)
        ImpReg.fit()
        warped_image, warped_mask, field = ImpReg(output_shape=img_insp.shape[-2:],
                                                  image=img_exp_tensor, mask=mask_exp)

        warped_image = (warped_image*255).astype(np.uint8)
        warped_mask = (warped_mask*255).astype(np.uint8)
        img_insp = (img_insp*255).astype(np.uint8)

        runtime = time.time() - t
        if torch.cuda.is_available():
            peak_memory_allocated = torch.cuda.max_memory_allocated("cuda") 
            peak_memory_allocated = peak_memory_allocated / 1024 / 1024 / 1024
        else:
            peak_memory_allocated = 0

        image = Image.fromarray(warped_image)
        image.save(output_path)
        image = Image.fromarray(warped_mask)
        image.save(output_path_mask)

        apply_warp_transform_to_landmarks(ImpReg.network, image_name=f1.parent.name+".png",
                                          shape=img_insp.shape, output_xml_path=output_folder + 'fixed_landmarks_1.xml')
        apply_warp_transform_to_landmarks(ImpReg.network, image_name=f1.parent.name+".png",
                                          shape=img_insp.shape, output_xml_path=output_folder + 'fixed_landmarks_2.xml')

        njd = analyze_deformation_field_2d(field, warped_image)

        _mse = mse(img_insp, warped_image)
        _ssim = ssim(img_insp, warped_image, data_range=255)
        _mi = mi(img_insp, warped_image)
        _cc = np.corrcoef(img_insp.flatten(), warped_image.flatten())[0, 1]
        dsc = dice_coefficient(warped_mask, mask_insp)

        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                try:
                    results = json.load(f)
                except json.JSONDecodeError:
                    results = {}
        else:
            results = {}

        mse_list.append(_mse)
        ssim_list.append(_ssim)
        mi_list.append(_mi)
        cc_list.append(_cc)
        dice_list.append(dsc)
        njd_list.append(njd)
        vram_list.append(peak_memory_allocated)
        runtime_list.append(runtime)

        results[str(output_path)] = {
            "MSE": round(_mse, 6),
            "SSIM": round(_ssim, 6),
            "MI": round(_mi, 6),
            "CC": round(_cc, 6),
            "DSC": round(dsc, 6),
            "NJD": round(njd, 6),
            "VRAM": round(peak_memory_allocated, 6),
            "Runtime": round(runtime, 6)
        }

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

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
                            "NJD": round(np.mean(njd_list), 10),
                            "VRAM": round(np.mean(vram_list), 10),
                            "Runtime": round(np.mean(runtime_list), 6)
                        }

with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Mean MSE: {np.mean(mse_list):.4f}")
print(f"Mean SSIM: {np.mean(ssim_list):.4f}")
print(f"Mean Mutual Information: {np.mean(mi_list):.4f}")
print(f"Mean Correlation Coefficient: {np.mean(cc_list):.4f}")
print(f"Mean dice: {np.mean(dice_list):.4f}")
print(f"Mean folded: {np.mean(njd_list):.10f}")
print(f"Mean vram: {np.mean(vram_list):.4f}")
print(f"Mean runtime: {np.mean(runtime_list):.4f}")
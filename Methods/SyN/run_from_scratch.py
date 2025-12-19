import os
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1" 
os.environ['TMPDIR'] = '/home/s.krasnova/tmp'

import ants
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import shutil
import xml.etree.ElementTree as ET

from skimage.metrics import mean_squared_error  as mse
from skimage.metrics import normalized_mutual_information as mi
from skimage.metrics import structural_similarity as ssim

from rTRE import calc_tre


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

def apply_transform_to_landmarks(warp_path, affine_path, image_name, output_xml_path):
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
                transformlist=[warp_path, affine_path]
            ) 
            for i, points_tag in enumerate(index_to_tag):
                new_x = transformed_points['x'].iloc[i]
                new_y = transformed_points['y'].iloc[i]
                points_tag.set('points', f"{new_y:.2f},{new_x:.2f}")

    tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)

def registration(before_img, after_img, moving_mask, fixed_mask, output_path,
                 f1_path, f2_path, p_mask, xml_path1, xml_path2, affine_path):
    fixed_img = after_img
    moving_img = before_img

    fixed_ants = ants.from_numpy(fixed_img)
    moving_ants = ants.from_numpy(moving_img)

    moving_ants = ants.apply_transforms(
        fixed=fixed_ants,
        moving=moving_ants,
        transformlist=[affine_path],
        interpolator='linear'
    )

    reg = ants.registration(
        fixed=fixed_ants,
        moving=moving_ants,
        type_of_transform="SyNOnly",
        random_seed=7
    )

    warped_img = ants.apply_transforms(
        fixed=fixed_ants,
        moving=moving_ants,
        transformlist=reg['fwdtransforms'],
        interpolator='linear'
    )

    warped_np = warped_img.numpy()
    warped = warped_np.astype(np.uint8)
    warped = Image.fromarray(warped)
    warped.save(output_path) 

    img1 = Image.fromarray(before_img)
    img2 = Image.fromarray(after_img)
    img1.save(f1_path)
    img2.save(f2_path)

    mask_ants = ants.from_numpy(moving_mask)
    warped_mask = ants.apply_transforms(
        fixed=fixed_ants,
        moving=mask_ants,
        transformlist=[reg['fwdtransforms'][0], affine_path],
        interpolator='genericLabel',
        verbose=False
    )

    mask = warped_mask.numpy()
    image = Image.fromarray((mask*255).astype(np.uint8))
    image.save(p_mask)

    njd = analyze_deformation_field_2d(reg['fwdtransforms'][0])

    image_name = f1_path.parent.name
    apply_transform_to_landmarks(reg['fwdtransforms'][0], affine_path, image_name + '.png', xml_path1)
    apply_transform_to_landmarks(reg['fwdtransforms'][0], affine_path, image_name + '.png', xml_path2)
    warped = np.array(warped)
    return mse(after_img, warped), ssim(after_img, warped, data_range=255),\
           mi(after_img, warped), np.corrcoef(after_img.flatten(), warped.flatten())[0, 1], \
           dice_coefficient(mask, fixed_mask), njd

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

def main():
    data_path = "Dataset/MGRegBench/"
    output_folder = "outputs/SyN/"
    os.makedirs(output_folder, exist_ok=True)
    shutil.copy2(data_path + 'moving_landmarks.xml', output_folder + 'moving_landmarks.xml')
    shutil.copy2(data_path + 'fixed_landmarks_1.xml', output_folder + 'fixed_landmarks_1.xml')
    shutil.copy2(data_path + 'fixed_landmarks_2.xml', output_folder + 'fixed_landmarks_2.xml')
    process_annotations(output_folder + 'moving_landmarks.xml')
    process_annotations(output_folder + 'fixed_landmarks_1.xml')
    process_annotations(output_folder + 'fixed_landmarks_2.xml')

    base_dir = Path("Dataset/MGRegBench/evaluation")
    masks_dir = Path("Dataset/MGRegBench/evaluation-masks")
    subdatasets = ["INBreast","KAU-BCMD", "RSNA"]
    json_file = output_folder + 'SyN.json'
    mse_list = []
    ssim_list = []
    mi_list = []
    cc_list = []
    dice_list = []
    njd_list = []

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

            affine_path = subdataset_path / patient_id / "affine.mat"
            warp_path = subdataset_path / patient_id / "warp.nii.gz"

            print(f"\n[Patient: {patient_id}]")
            before_img = Image.open(f1).convert('L')
            after_img = Image.open(f2).convert('L')

            min_height = min(before_img.height, after_img.height)
            min_width = min(before_img.width, after_img.width)
                
            if min_height > 1000:
                before_img = before_img.resize((min_width // 2, min_height // 2),
                                               Image.Resampling.LANCZOS)
                after_img = after_img.resize((min_width // 2, min_height // 2),
                                             Image.Resampling.LANCZOS)

            before_arr = np.array(before_img)
            after_arr = np.array(after_img)

            output_dir = Path(output_folder) / subdataset / patient_id
            output_dir.mkdir(parents=True, exist_ok=True)
            f1_path = output_dir / f"{f1.stem}.png"
            f2_path = output_dir / f"{f2.stem}.png"

            m1 = Image.open(masks_dir / subdataset / patient_id / f"{f1.stem}.png").convert('L')
            m2 = Image.open(masks_dir / subdataset / patient_id / f"{f2.stem}.png").convert('L')

            if min_height > 1000:
                m1 = m1.resize((min_width // 2, min_height // 2), Image.Resampling.LANCZOS)
                m2 = m2.resize((min_width // 2, min_height // 2), Image.Resampling.LANCZOS)

            m1_arr = np.array(m1)
            m2_arr = np.array(m2)
            m1_arr = (m1_arr > 180).astype(np.uint8)
            m2_arr = (m2_arr > 180).astype(np.uint8)

            p = output_dir / "mask.png"

            if subdataset == "RSNA":
                output_path = output_dir / f"{f1.stem}_{f2.stem}.png"
                before_arr = before_arr * m1_arr
                after_arr = after_arr * m2_arr
            else:
                name = merge_two_filenames(f1.stem, f2.stem)
                output_path = output_dir / f"{name}.png"

            try:
                mse, ssim, mi, cc, dice, njd = registration(before_arr, after_arr,
                                                            m1_arr, m2_arr, output_path,
                                                            f1_path, f2_path, p,
                                                            output_folder + 'fixed_landmarks_1.xml',
                                                            output_folder + 'fixed_landmarks_2.xml', str(affine_path))

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
                    "NJD": round(njd, 6) 
                }
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

                mse_list.append(mse)
                ssim_list.append(ssim)
                mi_list.append(mi)
                cc_list.append(cc)
                dice_list.append(dice)
                njd_list.append(njd)

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
                            "NJD": round(np.mean(njd_list), 6) 
                        }

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Mean MSE: {np.mean(mse_list):.4f}")
    print(f"Mean SSIM: {np.mean(ssim_list):.4f}")
    print(f"Mean Mutual Information: {np.mean(mi_list):.4f}")
    print(f"Mean Correlation Coefficient: {np.mean(cc_list):.4f}")
    print(f"Mean DSC: {np.mean(dice_list):.4f}")
    print(f"Mean NJD: {np.mean(njd_list):.4f}")

    calc_tre(output_folder + 'moving_landmarks.xml',
             output_folder + 'fixed_landmarks_1.xml',
             output_folder + 'fixed_landmarks_2.xml')


if __name__ == "__main__":
    main()
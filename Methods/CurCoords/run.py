import cv2
import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
from PIL import Image
from scipy.ndimage import map_coordinates
import shutil
import time
import xml.etree.ElementTree as ET

from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_mutual_information as mi
from skimage.metrics import structural_similarity as ssim

from analysis import filter_points_inside_hull, find_nearest_points
from CurvilinearCoordinates import CurvilinearCoordinatesBuilder


def analyze_deformation_field_2d(reg1, reg2, s_coords, t_coords):
    J1 = reg1.get_J(s_coords, t_coords)
    J2 = reg2.get_J(s_coords, t_coords)
    dx_ds = J1[:, 0, 0]
    dx_dt = J1[:, 0, 1]
    dy_ds = J1[:, 1, 0]
    dy_dt = J1[:, 1, 1]
    jacobian_det_1 = dx_ds * dy_dt - dx_dt * dy_ds

    dx_ds = J2[:, 0, 0]
    dx_dt = J2[:, 0, 1]
    dy_ds = J2[:, 1, 0]
    dy_dt = J2[:, 1, 1]
    jacobian_det_2 = dx_ds * dy_dt - dx_dt * dy_ds

    jacobian_det = jacobian_det_2 / (jacobian_det_1 + 1e-6)
    percent_folded_voxels = 100 * np.sum(jacobian_det < 0) / jacobian_det.size
    return percent_folded_voxels

def dice_coefficient(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2).sum()
    size1 = mask1.sum()
    size2 = mask2.sum()
    dice = (2.0 * intersection) / (size1 + size2)
    return dice

def registration(
    registrator1,
    registrator2,
    orientation,
    get_dice=False
):
    s_values = np.linspace(0, 1, 10)
    t_values = np.linspace(0, 1, 10)

    grid_s, grid_t = np.meshgrid(s_values, t_values, indexing='ij')
    cartesian_x, cartesian_y = registrator1.curvilinear_to_cartesian(grid_s, grid_t)

    cartesian_x = cartesian_x + registrator1.start_system[0]
    cartesian_y = cartesian_y + registrator1.start_system[1]

    points_inside, points_outside, hull, hull_path = filter_points_inside_hull(
        registrator1.mask, cartesian_x, cartesian_y
    )

    s_values = np.linspace(0, 1, 500)
    t_values = np.linspace(0, 1, 500)

    grid_s, grid_t = np.meshgrid(s_values, t_values, indexing='ij')
    cartesian_x, cartesian_y = registrator1.curvilinear_to_cartesian(grid_s, grid_t)

    cartesian_x = cartesian_x + registrator1.start_system[0]
    cartesian_y = cartesian_y + registrator1.start_system[1]

    s_coords, t_coords = find_nearest_points(
        points_inside, cartesian_x, cartesian_y, grid_s, grid_t
    )

    cartesian_x_2, cartesian_y_2 = registrator2.curvilinear_to_cartesian(s_coords, t_coords)

    cartesian_y_2 += registrator2.start_system[1]
    cartesian_x_2 += registrator2.start_system[0]

    x_coords = np.array(cartesian_x_2)
    y_coords = np.array(cartesian_y_2)

    coordinates = np.vstack((y_coords, x_coords))

    interpolated_values = map_coordinates(
        registrator2.mammogram,
        coordinates,
        order=1,
        mode='constant',
        cval=0.0
    )

    original_image = registrator2.mammogram
    output_image = np.zeros_like(original_image)

    x_coords = points_inside[:, 0]
    y_coords = points_inside[:, 1]

    output_image[y_coords, x_coords] = interpolated_values

    output_mask = None
    if get_dice:
        output_mask = np.zeros_like(original_image)
        interpolated_values = map_coordinates(
            registrator2.mask,
            coordinates,
            order=1,
            mode='constant',
            cval=0.0
        )
        output_mask[y_coords, x_coords] = interpolated_values

    if orientation == 'L':
        return np.flip(output_image, axis=1), s_coords, t_coords, output_mask
    else:
        return output_image, s_coords, t_coords, output_mask

def landmark_error(registrator1, registrator2, orientation, image_name, output_xml_path):
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
                if orientation == 'L':
                    x = w - x - 1
                point_list.append([int(round(x)), int(round(y))])
                index_to_tag.append(points_tag)
            
            points = np.array(point_list)

            s_values = np.linspace(0, 1, 500)
            t_values = np.linspace(0, 1, 500)

            grid_s, grid_t = np.meshgrid(s_values, t_values, indexing='ij')
            cartesian_x, cartesian_y = registrator1.curvilinear_to_cartesian(grid_s, grid_t)

            cartesian_x = cartesian_x + registrator1.start_system[0]
            cartesian_y = cartesian_y + registrator1.start_system[1]
            
            s_coords, t_coords = find_nearest_points(
                points, cartesian_x, cartesian_y, grid_s, grid_t
            )

            cartesian_x_2, cartesian_y_2 = registrator2.curvilinear_to_cartesian(s_coords, t_coords)

            cartesian_y_2 += registrator2.start_system[1]
            cartesian_x_2 += registrator2.start_system[0]

            for i, points_tag in enumerate(index_to_tag):
                new_y, new_x = cartesian_y_2[i], cartesian_x_2[i]
                if orientation == 'L':
                    new_x = w - new_x - 1
                points_tag.set('points', f"{new_x:.2f},{new_y:.2f}")

    tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)

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
    

data_path = "Dataset/MGRegBench/"
out_folder = "outputs/CurCoords/"
os.makedirs(out_folder, exist_ok=True)
shutil.copy2(data_path + 'moving_landmarks.xml', out_folder + 'moving_landmarks.xml')
shutil.copy2(data_path + 'fixed_landmarks_1.xml', out_folder + 'fixed_landmarks_1.xml')
shutil.copy2(data_path + 'fixed_landmarks_2.xml', out_folder + 'fixed_landmarks_2.xml')
process_annotations(out_folder + 'moving_landmarks.xml')
process_annotations(out_folder + 'fixed_landmarks_1.xml')
process_annotations(out_folder + 'fixed_landmarks_2.xml')

base_images = Path("Dataset/MGRegBench/evaluation")
base_masks  = Path("Dataset/MGRegBench/evaluation-masks")
csv_path = base_images / "metadata_evaluation.csv"

df = pd.read_csv(csv_path)

df["File Path"] = df["File Path"].astype(str).apply(os.path.normpath)
metadata_dict = {
    row["File Path"]: row["Laterality"]
    for _, row in df.iterrows()
}

datasets = ["INBreast", "KAU-BCMD", "RSNA"]

json_file = out_folder + 'CurCoords.json'

mse_list = []
ssim_list = []
mi_list = []
cc_list = []
dice_list = []
njd_list = []
runtime_list = []

for dataset in datasets:
    img_dataset_path = base_images / dataset
    mask_dataset_path = base_masks / dataset

    print(f"\n{'='*50}")
    print(f"PROCESSING: {dataset}")
    print(f"{'='*50}")

    for subdir in img_dataset_path.iterdir():
        if not subdir.is_dir():
            continue
        print(f"\n[Patient: {subdir.name}]")
        mask_subdir = mask_dataset_path / subdir.name

        img_files = []
        for ext in ["*.jpg", "*.png"]:
            img_files.extend(subdir.glob(ext))

        mask_files = [
            mask_subdir / f"{img_file.stem}.png"
            for img_file in img_files
        ]
        img_path1 = img_files[0]
        img_path2 = img_files[1]
        before = Image.open(img_path1).convert('L')
        after = Image.open(img_path2).convert('L')

        after_np = np.array(after)
        before_np = np.array(before)
        height, width = after_np.shape[:2]
        if height > 1000:
            height = height // 2
            width = width // 2
            after_np = cv2.resize(after_np, (width, height), interpolation=cv2.INTER_LINEAR)

        mask_path1 = str(mask_files[0].resolve())
        mask_path2 = str(mask_files[1].resolve())
        norm_img1 = os.path.normpath(img_path1)

        orientation = metadata_dict[norm_img1]
        mut_inf = 0
        start_point = [0.3, 0.3]
        reg_result = None
        t = time.time()
        for i in np.arange(0.3, 0.71, 0.04):
            for j in np.arange(0.3, 0.71, 0.04):
                registrator1 = CurvilinearCoordinatesBuilder(i, orientation,
                                                             mask_path=mask_path2,
                                                             mammogram_path=img_path2)
                registrator2 = CurvilinearCoordinatesBuilder(j, orientation,
                                                             mask_path=mask_path1,
                                                             mammogram_path=img_path1)

                out, _, _, _ = registration(
                    registrator1,
                    registrator2,
                    orientation
                    )
                mi_local = mi(out, after_np)
                if mi_local > mut_inf:
                    start_point[0] = i
                    start_point[1] = j
                    reg_result = out
                    mut_inf = mi_local

        registrator1 = CurvilinearCoordinatesBuilder(start_point[0], orientation,
                                                     mask_path=mask_path2, mammogram_path=img_path2)
        registrator2 = CurvilinearCoordinatesBuilder(start_point[1], orientation,
                                                     mask_path=mask_path1, mammogram_path=img_path1)
        out, s_coords, t_coords, mask_warped = registration(
                    registrator1,
                    registrator2,
                    orientation,
                    get_dice=True
                    )
        landmark_error(registrator2, registrator1,
                       orientation,
                       Path(img_path1).parent.name + ".png",
                       out_folder + 'moving_landmarks.xml')
        runtime = time.time() - t
        njd = analyze_deformation_field_2d(registrator1, registrator2, s_coords, t_coords)
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                try:
                    results = json.load(f)
                except json.JSONDecodeError:
                    results = {}
        else:
            results = {}

        mse_list.append(mse(after_np, reg_result))
        ssim_list.append(ssim(after_np, reg_result, data_range=255))
        mi_list.append(mi(after_np, reg_result))
        cc_list.append(np.corrcoef(after_np.flatten(), reg_result.flatten())[0, 1])
        dice_list.append(dice_coefficient(registrator1.mask, mask_warped))
        njd_list.append(njd)
        runtime_list.append(runtime)

        p = mask_subdir
        last_three = Path(*p.parts[-2:])
        output_path = os.path.join(out_folder, last_three)
        Path(output_path).mkdir(parents=True, exist_ok=True)

        output_path_reg = os.path.join(output_path, "moved.png")
        Image.fromarray(reg_result).save(output_path_reg)

        output_path_1 = os.path.join(output_path, f"{img_files[0].stem}.png")
        output_path_2 = os.path.join(output_path, f"{img_files[1].stem}.png")
        before.save(output_path_1)
        after.save(output_path_2)
        results[str(output_path_reg)] = {
                        "MSE": round(mse_list[-1], 6),
                        "SSIM": round(ssim_list[-1], 6),
                        "MI": round(mi_list[-1], 6),
                        "CC": round(cc_list[-1], 6),
                        "DSC": round(dice_list[-1], 6),
                        "NJD": round(njd_list[-1], 6),
                        "Runtime": round(runtime_list[-1], 6)
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
                            "NJD": round(np.mean(njd_list), 6),
                            "Runtime": round(np.mean(runtime_list), 6)
                        }

with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

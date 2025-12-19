import os
import argparse
import shutil
import pandas as pd
from pathlib import Path
from PIL import Image
import SimpleITK as sitk
import numpy as np

def convert_dcm_to_png(dcm_path, png_path):
    ds = sitk.ReadImage(dcm_path)
    img = sitk.GetArrayFromImage(ds)
    img = np.squeeze(img)
    img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(png_path, format="PNG")

def copy_rsna_images(metadata_file, source_dir, target_base_dir):    
    df = pd.read_csv(metadata_file)
    
    rsna_rows = df[df['File Path'].str.contains('RSNA', na=False)]
        
    Path(target_base_dir).mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    converted_count = 0
    errors = []
    
    for idx, row in rsna_rows.iterrows():
        file_path = row['File Path']
        
        path_parts = file_path.split('/')
        patient_folder_with_suffix = path_parts[-2]
        filename_png = path_parts[-1]
        filename_dcm = filename_png.replace('.png', '.dcm')
        
        patient_id = patient_folder_with_suffix.split('-')[0]
        
        source_dcm_path = Path(source_dir) / patient_id / filename_dcm
        print(source_dcm_path)
        
        target_rel_dir = Path('RSNA') / patient_folder_with_suffix
        target_full_dir = Path(target_base_dir) / target_rel_dir
        target_dcm_path = target_full_dir / filename_dcm
        target_png_path = target_full_dir / filename_png
        
        if not source_dcm_path.exists():
            errors.append(f"Source file hasn't found: {source_dcm_path}")
            continue
        
        target_full_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            shutil.copy2(source_dcm_path, target_dcm_path)
            copied_count += 1
            
            if convert_dcm_to_png(target_dcm_path, target_png_path):
                converted_count += 1
                
            print(f"Done : {patient_folder_with_suffix}/{filename_png}")
            
        except Exception as e:
            errors.append(f"Error during {source_dcm_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare RSNA data for MGRegBench")
    parser.add_argument("--rsna-root", type=str, required=True,
                        help="Path to RSNA dataset folder (e.g., /home/rsna-breast-cancer-detection)")

    args = parser.parse_args()
    source_dir = args.rsna_root + '/train_images'

    metadata_file = "Dataset/MGRegBench/train/metadata_train.csv"
    target_base_dir = "Dataset/MGRegBench/train"
    copy_rsna_images(metadata_file, source_dir, target_base_dir)

    metadata_file = "Dataset/MGRegBench/evaluation/metadata_evaluation.csv"
    target_base_dir = "Dataset/MGRegBench/evaluation"
    copy_rsna_images(metadata_file, source_dir, target_base_dir)

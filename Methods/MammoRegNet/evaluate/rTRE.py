import xml.etree.ElementTree as ET
import math
import argparse
from collections import defaultdict

def parse_annotations(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    data = {}

    for image in root.findall('image'):
        name = image.get('name')
        width = int(image.get('width'))
        height = int(image.get('height'))
        points_dict = {}
        for points in image.findall('points'):
            attr = points.find('attribute[@name="id"]')
            if attr is None:
                continue
            try:
                point_id = int(attr.text)
            except ValueError:
                continue

            coords_str = points.get('points')
            if not coords_str:
                continue
            try:
                x, y = map(float, coords_str.split(','))
                points_dict[point_id] = (x, y)
            except (ValueError, IndexError):
                continue

        data[name] = {
            'width': width,
            'height': height,
            'points': points_dict
        }
    return data

def compute_average_normalized_distance(img_data_left, img_data_other):
    points_left = img_data_left['points']
    points_other = img_data_other['points']
    common_ids = set(points_left.keys()) & set(points_other.keys())
    
    if not common_ids:
        return 0.0, 0

    w = img_data_other['width']
    h = img_data_other['height']
    diagonal = math.sqrt(w * w + h * h)

    total_dist = 0.0
    for pid in common_ids:
        x1, y1 = points_left[pid]
        x2, y2 = points_other[pid]
        dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        total_dist += dist / diagonal

    return total_dist / len(common_ids), len(common_ids)

def calc_tre(left_file, right_file, validated_file):
    print("Calculating rTRE...")
    left_data = parse_annotations(left_file)
    right_data = parse_annotations(right_file)
    validated_data = parse_annotations(validated_file)

    all_image_names = set(left_data.keys()) & set(right_data.keys()) & set(validated_data.keys())
    per_image_scores = []

    for name in sorted(all_image_names):
        img_left = left_data[name]
        img_right = right_data[name]
        img_valid = validated_data[name]

        avg_lr, count_lr = compute_average_normalized_distance(img_left, img_right)
        avg_lv, count_lv = compute_average_normalized_distance(img_left, img_valid)

        if count_lr == 0 and count_lv == 0:
            continue
        elif count_lr == 0:
            combined = avg_lv
        elif count_lv == 0:
            combined = avg_lr
        else:
            combined = (avg_lr + avg_lv) / 2.0
        #print(combined)
        per_image_scores.append(combined)

    final_score = sum(per_image_scores) / len(per_image_scores)
    print(f"\nFinal average normalized distance: {final_score:.6f}")

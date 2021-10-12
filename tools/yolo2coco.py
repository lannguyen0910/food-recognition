import numpy as np
import json
import os
import argparse
from tqdm import tqdm
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

idx_classes = [
    'Aortic enlargement',
    'Atelectasis',
    'Calcification',
    'Cardiomegaly',
    'Consolidation',
    'ILD',
    'Infiltration',
    'Lung Opacity',
    'Nodule/Mass',
    'Other lesion',
    'Pleural effusion',
    'Pleural thickening',
    'Pneumothorax',
    'Pulmonary fibrosis',
    'No Finding'
]
img_w, img_h = (1024, 1024)


def convert(args):

    img_path = args.img_path
    ann_path = args.ann_path
    out_path = args.out_path

    my_dict = {
        'images': [],
        'annotations': [],
        'categories': []
    }

    for idx, cls in enumerate(idx_classes):
        cls_dict = {
            'supercategory': 'X-Ray',
            'id': idx+1,
            'name': cls
        }
        my_dict['categories'].append(cls_dict)

    paths = os.listdir(ann_path)
    ann_idx = 1
    for path in tqdm(paths):
        frame_name = path[:-4]
        img_idx = "".join(frame_name.split('_')[1:])
        ann_name = os.path.join(ann_path, frame_name+'.txt')
        img_name = os.path.join(img_path, frame_name+'.jpg')

        with open(ann_name, 'r') as f:
            data = f.read()
            for line in data.splitlines():
                item = line.split()
                cls = int(item[0])
                w = float(item[3]) * img_w
                h = float(item[4]) * img_h
                x = float(item[1]) * img_w - w/2
                y = float(item[2]) * img_h - h/2
                box = [x, y, w, h]

                ann_dict = {
                    'segmentation': [[]],
                    'area': 0,
                    'iscrowd': 0,
                    'image_id': int(img_idx),
                    'bbox': box,
                    'category_id': cls+1,
                    'id': ann_idx,
                    'ignore': 0
                }
                ann_idx += 1

                my_dict['annotations'].append(ann_dict)

        img_dict = {
            'file_name': frame_name+'.jpg',
            'height': img_h,
            'width': img_w,
            'id': int(img_idx)}

        my_dict['images'].append(img_dict)

    with open(out_path, 'w') as outfile:
        json.dump(my_dict, outfile)

    print('Saved to', out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert yolo format to coco format')
    parser.add_argument('--ann_path', type=str,
                        help='path to yolo format folder')
    parser.add_argument('--img_path', type=str, help='path to image folder')
    parser.add_argument('--out_path', type=str,
                        default='./annotations.json', help='path to output json file')
    args = parser.parse_args()
    convert(args)

import cv2
import os
from shutil import copyfile
import json
from tqdm import tqdm

ROOT = './data/vn_food/raw'
out_folder = './data/vn_food/images'
ann_path = './data/vn_food/annotations'
my_dict = {
    'images':[],
    'annotations': [],
    'categories':[]
}

class_names = [
"Banh_mi",
"Banh_trang_tron",
"Banh_xeo",
"Bun_bo_Hue",
"Bun_dau",
"Com_tam",
"Goi_cuon",
"Pho",
"Hu_tieu",
"Xoi"]

image_id = 0
obj_id = 0


for class_id, class_name in tqdm(enumerate(class_names)):

    cat_dict = {
        'id': class_id+1,
        'name': class_name,
        'supercategory': "food"
    }

    my_dict['categories'].append(cat_dict)

    image_folder = os.path.join(ROOT, class_name)
    file_names = os.listdir(image_folder)
    image_names = [i for i in file_names if not i.endswith('txt')]
    for image_name in image_names:
        image_path = os.path.join(image_folder,image_name)
        img = cv2.imread(image_path)
        height, width, _ = img.shape

        new_image_path = os.path.join(out_folder, image_name)
        copyfile(image_path, new_image_path)
        img_dict = {
            "id": image_id,
            "file_name": image_name,
            "height": height,
            "width": width
        }

        my_dict['images'].append(img_dict)


        _, ext = os.path.splitext(image_name)
        ext_len = str.__len__(ext)
        text_name = image_path[:-ext_len]+ '.txt'
        with open(text_name, 'r') as f:
            data = f.read()
            lines = data.splitlines()
        for line in lines:
            item = line.split(' ')
            box = [float(i) for i in item[1:]]
            box[0] -= box[2]/2
            box[1] -= box[3]/2

            box[0] = int(box[0] * width)
            box[1] = int(box[1] * height)
            box[2] = int(box[2] * width)
            box[3] = int(box[3] * height)

            box_dict ={
                "id": obj_id,
                "bbox": box,
                "area": box[2]*box[3],
                "category_id": class_id+1,
                "image_id": image_id,
                "iscrowd": 0
            }

            my_dict['annotations'].append(box_dict)
            obj_id+=1
        image_id+=1

    
with open(f'{ann_path}/annotations.json', 'w+') as f:
    json.dump(my_dict, f)



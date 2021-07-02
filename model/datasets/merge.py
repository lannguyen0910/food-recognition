import json
from tqdm import tqdm


def get_all_boxes_of_image(image_id, annotations):
    boxes = []
    for box in annotations:
        if box['image_id'] == image_id:
            boxes.append(box)
    return boxes

def merge(list_of_dict):
    my_dict = {
        'images':[],
        'annotations':[],
        'categories': []
    }

    # Init
    image_id = 0
    obj_id = 0
    cat_idx = 1

    # Make categories
    category_dict = {}
    for dict_ in list_of_dict:
        cat_list = dict_['categories']
        idx_category_map = {}
        idx_image_map = {}

        for cat in cat_list:
            cat_name = cat['name']
            old_cat_id = cat['id']
            if cat_name not in category_dict.keys():
                category_dict[cat_name] = cat_idx
                idx_category_map[old_cat_id] = cat_idx
                cat_idx += 1
            else:
                idx_category_map[old_cat_id] = category_dict[cat_name]

        image_list = dict_['images']
        for image in tqdm(image_list):
            old_image_id = image['id']
            image_name = image['file_name']
            width = image['width']
            height = image['height']

            idx_image_map[old_image_id]=image_id

            image_dict = {
                'id': image_id,
                'file_name': image_name,
                'width': width,
                'height': height
            }

            my_dict['images'].append(image_dict)
            
        
            boxes = get_all_boxes_of_image(old_image_id, dict_['annotations'])

            for box in boxes:
                obj_dict = {
                    "id": obj_id, 
                    "image_id": image_id,
                    "bbox": box['bbox'],
                    "category_id": idx_category_map[box['category_id']], 
                    "area": box['bbox'][2]*box['bbox'][3], 
                    "iscrowd": 0
                }

                my_dict['annotations'].append(obj_dict)
                obj_id+=1

            image_id+=1
    
    for key in category_dict.keys():
        cat_dict = {
            'id': category_dict[key],
            'name': key,
            'supercategory': 'food'
        }

        my_dict['categories'].append(cat_dict)

    with open('val.json', 'w') as f:
        json.dump(my_dict, f)

    

if __name__ == '__main__':
    with open('./data/vn_food/annotations/annotations_val.json', 'r') as f:
        dict1 = json.load(f) 

    with open('./data/school_lunch/annotations/val.json', 'r') as f:
        dict2 = json.load(f) 

    with open('./data/open-images-food/annotations/val.json', 'r') as f:
        dict3 = json.load(f) 

    merge([dict1, dict2, dict3])
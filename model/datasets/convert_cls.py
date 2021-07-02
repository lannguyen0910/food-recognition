import os
from shutil import copyfile
from tqdm import tqdm




def convert_food101():
    ROOT = './data/MAFood121/images'
    TRAIN_SPLIT = './data/MAFood121/annotations/train.txt'
    VAL_SPLIT = './data/MAFood121/annotations/test.txt'
    CLASS_ID = './data/MAFood121/annotations/dishes.txt'
    
    def get_label_image_name(path):
        return path.split('/')[0], path.split('/')[1]
    
    def split(name='train', class_mapping=None):
        if name == 'train':
            SPLIT = TRAIN_SPLIT
        else:
            SPLIT = VAL_SPLIT

        with open(SPLIT, 'r') as f:
            data = f.read()
            lines = data.splitlines()

       
        new_root = f'./data/MAFood121/new_images'
        for line in tqdm(lines):
            label, image_name = get_label_image_name(line)
            old_path = os.path.join(ROOT, line)
            
            label = class_mapping[label]
            new_folder = os.path.join(new_root, name, label)
            os.makedirs(new_folder, exist_ok=True)

            new_path = os.path.join(new_folder, image_name)
            copyfile(old_path, new_path)
    
    # Start splitting

    with open(CLASS_ID, 'r') as f:
        class_ids = f.read().splitlines()
    
    class_mapping = {k[2:]:k[2:].replace('_',' ').capitalize() for k in class_ids}

    print(class_mapping)
    for splits in ['train', 'val']:
        split(splits, class_mapping)


if __name__ == '__main__':
    convert_food101()


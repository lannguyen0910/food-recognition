import os
from tqdm import tqdm
import cv2
import xml.etree.ElementTree as ET

TRAIN_PATH = 'data/train/'
VALID_PATH = 'data/val/'

# get list file
train_file = list({i[:-3] for i in os.listdir(TRAIN_PATH) if i[-3:] == 'xml'})
valid_file = list({i[:-3] for i in os.listdir(VALID_PATH) if i[-3:] == 'xml'})

print("# of train file:", len(train_file))
print("# of valid file:", len(valid_file))


def convert(files, file_path):
    pbar = tqdm(range(len(files)))
    for i in pbar:
        PATH = f'{file_path}{files[i]}'
        tree = ET.parse(PATH+'xml')
        root = tree.getroot()

        s = ""

        try:
            img = cv2.imread(PATH+'jpg')
            height, width = img.shape[:2]
        except:
            img = cv2.imread(PATH+'png')
            height, width = img.shape[:2]

        for element in root.iter('object'):
            x_min, x_max, y_min, y_max = 0, 0, 0, 0
            i_class = None
            for sub_element in element.iter('name'):
                if sub_element.text == 'face':
                    i_class = 0
                else:
                    i_class = 1
            for sub_element in element.iter('bndbox'):
                for index in sub_element:
                    if index.tag == 'xmin':
                        x_min = int(index.text)
                    if index.tag == 'ymin':
                        y_min = int(index.text)
                    if index.tag == 'xmax':
                        x_max = int(index.text)
                    if index.tag == 'ymax':
                        y_max = int(index.text)
                x = round((x_max/2 + x_min/2)/width, 6)
                y = round((y_max/2 + y_min/2)/height, 6)
                box_width = round((x_max - x_min)/width, 6)
                box_height = round((y_max - y_min)/height, 6)

                res = [i_class, x, y, box_width, box_height]
                for e in res:
                    s += str(e) + ' '
            s += '\n'

        with open(PATH+'txt', 'w') as f:
            f.write(s)
            f.close()
            # print(i_class, x, y, box_width, box_height)

        # remove file
        os.remove(PATH+'xml')


# processing xml to yolo format txt
convert(train_file, TRAIN_PATH)
convert(valid_file, VALID_PATH)

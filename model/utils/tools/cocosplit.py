import json
import argparse
import funcy
from sklearn.model_selection import train_test_split

"""
Example: python cocosplit.py --having-annotations --ratio=0.8 --annotations=/path/to/your/coco_annotations.json --train=train.json --test=test.json
"""


parser = argparse.ArgumentParser(description='Splits COCO annotations file into training and test sets.')
parser.add_argument('--annotations', dest='annotations', metavar='coco_annotations', type=str,
                    help='Path to COCO annotations file.')
parser.add_argument('--ratio', dest='ratio', type=float, required=True,
                    help="A percentage of a split; a number in (0, 1)")
parser.add_argument('--having-annotations', dest='having_annotations', action='store_true',
                    help='Ignore all images without annotations. Keep only these with at least one annotation')

args = parser.parse_args()
TRAIN_PATH = args.annotations[:-5]+'_train.json'
VAL_PATH = args.annotations[:-5]+'_val.json'

def save_coco(file, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)

def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)

def main(args):
    with open(args.annotations, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
       
        images = coco['images'] 
        annotations = coco['annotations']
        categories = coco['categories']

        number_of_images = len(images)

        images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)

        if args.having_annotations:
            images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

        x, y = train_test_split(images, train_size=args.ratio)

        save_coco(TRAIN_PATH,  x, filter_annotations(annotations, x), categories)
        save_coco(VAL_PATH, y, filter_annotations(annotations, y), categories)

        print('Split completed!')


if __name__ == "__main__":
    main(args)
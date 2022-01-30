from .utils.getter import *
import argparse
import os
import torch
from torch.utils.data import  DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import pandas as pd
from tqdm import tqdm
from .augmentations.transforms import get_resize_augmentation
from .augmentations.transforms import MEAN, STD

parser = argparse.ArgumentParser(description='Classify an image / folder of images')
parser.add_argument('--weight', type=str ,help='trained weight')
parser.add_argument('--input_path', type=str, help='path to an image to inference')
parser.add_argument('--output_path', type=str, help='path to save csv result file')

# Global model, only changes when model name changes
CLASSIFIER = None

class ClassificationTestset():
    def __init__(self, config, img_list):
        self.img_list = img_list # list of cv2 images

        self.transforms = A.Compose([
            get_resize_augmentation(config.image_size, keep_ratio=config.keep_ratio),
            A.Normalize(mean=MEAN, std=STD, max_pixel_value=1.0, p=1.0),
            ToTensorV2(p=1.0)
        ])

    def __getitem__(self, idx):
        img = self.img_list[idx]
        img = img.astype(np.float32)
        img /= 255.0
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        return {
            'img': img,
        }

    def collate_fn(self, batch):
        imgs = torch.stack([s['img'] for s in batch])  
        return {
            'imgs': imgs
        }

    def __len__(self):
        return len(self.img_list)

    def __str__(self):
        return f"Number of found images: {len(self.img_list)}"
  
def classify(weight, img_list):
    global CLASSIFIER
    config = get_config(weight)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    testset = ClassificationTestset(config, img_list)
    testloader = DataLoader(
        testset,
        batch_size=1,
        num_workers=2,
        pin_memory=True,
        collate_fn=testset.collate_fn
    )

    class_names, num_classes = get_class_names(weight)

    if CLASSIFIER is None or CLASSIFIER.model_name != config.model_name:
        net = BaseTimmModel(
            name=config.model_name, 
            num_classes=num_classes)
        CLASSIFIER = Classifier( model = net,  device = device, freeze=True)
        load_checkpoint(CLASSIFIER, weight)

        ## Print info
        print(config)

    CLASSIFIER.eval()

    pred_list = []
    prob_list = []

    with tqdm(total=len(testloader)) as pbar:
        with torch.no_grad():
            for idx, batch in enumerate(testloader):
                preds, probs = CLASSIFIER.inference_step(batch, return_probs=True)
                for idx, (pred, prob) in enumerate(zip(preds, probs)):
                    pred_list.append(class_names[pred])
                    prob_list.append(prob)

    return pred_list, prob_list
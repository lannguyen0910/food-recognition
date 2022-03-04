from typing import Iterable, List
import os
from PIL import Image
import torch
import torch.utils.data as data

class ConcatDataset(data.ConcatDataset):
    """
    Concatenate dataset and do sampling randomly

    datasets: `Iterable[data.Dataset]`
        list of datasets
    """
    def __init__(self, datasets: Iterable[data.Dataset], **kwargs) -> None:
        super().__init__(datasets)

        # Workaround, not a good solution
        self.classnames = datasets[0].classnames
        self.collate_fn = datasets[0].collate_fn


class ChainDataset(data.ConcatDataset):
    """
    Chain dataset and do sampling iteratively

    datasets: `Iterable[data.Dataset]`
        list of datasets
    """
    def __init__(self, datasets: Iterable[data.Dataset], **kwargs) -> None:
        super().__init__(datasets)

        # Workaround, not a good solution
        self.classnames = datasets[0].classnames
        self.collate_fn = datasets[0].collate_fn

class ImageDataset(data.Dataset):
    """
    Dataset contains folder of images 

    image_dir: `str`
        path to folder of images
    txt_classnames: `str`
        path to .txt file contains classnames
    transform: `List`
        list of transformation
    """
    def __init__(self, image_dir: str, txt_classnames:str, transform: List =None, **kwargs):
        super().__init__()
        self.image_dir = image_dir
        self.txt_classnames = txt_classnames
        self.transform = transform
        self.load_data()

    def load_data(self):
        """
        Load filepaths into memory
        """
        with open(self.txt_classnames, 'r') as f:
            self.classnames = f.read().splitlines()
        self.fns = []
        image_names = os.listdir(self.image_dir)
        for image_name in image_names:
            self.fns.append(image_name)

    def __getitem__(self, index: int):
        """
        Get an item from memory
        """
        image_name = self.fns[index]
        image_path = os.path.join(self.image_dir, image_name)
        im = Image.open(image_path).convert('RGB')
        width, height = im.width, im.height

        if self.transform is not None: 
            im = self.transform(im)

        return {
            "input": im, 
            'img_name': image_name,
            'ori_size': [width, height]
        }

    def __len__(self):
        return len(self.fns)

    def collate_fn(self, batch: List):
        imgs = torch.stack([s['input'] for s in batch])
        img_names = [s['img_name'] for s in batch]
        ori_sizes = [s['ori_size'] for s in batch]

        return {
            'inputs': imgs,
            'img_names': img_names,
            'ori_sizes': ori_sizes
        }
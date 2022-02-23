import os
import torch
from PIL import Image
from typing import Dict, List

class ClassificationDataset(torch.utils.data.Dataset):
    r"""Base dataset for classification tasks
    """

    def __init__(
        self,
        test: bool = False,
        **kwargs
    ):
        super(ClassificationDataset, self).__init__(**kwargs)
        self.train = not (test)
        self.classes_idx = {}
        self.classnames = None
        self.transform = None
        self.fns = [] # list of [filename, label]
        self.classes_dist = [] # Classes distribution (for balanced sampler)

    def _load_data(self):
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict:
        """
        Get one item
        """
        image_path, label_name = self.fns[idx]
        im = Image.open(image_path).convert('RGB')
        width, height = im.width, im.height
        class_idx = self.classes_idx[label_name]

        if self.transform:
            im = self.transform(im)

        target = {}
        target['labels'] = [class_idx]
        target['label_name'] = label_name

        return {
            "input": im, 
            'target': target,
            'img_name': os.path.basename(image_path),
            'ori_size': [width, height]
        }

    def __len__(self) -> int:
        return len(self.fns)

    def collate_fn(self, batch: List):
        """
        Collator for wrapping a batch
        """
        imgs = torch.stack([s['input'] for s in batch])
        targets = torch.stack([torch.LongTensor(s['target']['labels']) for s in batch])
        img_names = [s['img_name'] for s in batch]

        # if self.mixupcutmix is not None:
        #     imgs, targets = self.mixupcutmix(imgs, targets.squeeze(1))
        # targets = targets.float()

        return {
            'inputs': imgs,
            'targets': targets,
            'img_names': img_names
        }
# Author: Zylo117
import os
import timm
import numpy as np

import torch
from torch import nn

from .yolo import Yolov4, Model
from .loss import YoloLoss
from .utils import non_max_suppression
from utilities.utils.utils import download_pretrained_weights

CACHE_DIR = './.cache'


def get_model(args, config, num_classes):

    NUM_CLASSES = num_classes
    print('Number of classes: ', NUM_CLASSES)
    max_post_nms = config.max_post_nms if config.max_post_nms > 0 else None
    max_pre_nms = config.max_pre_nms if config.max_pre_nms > 0 else None

    net = None

    if args.weight is None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        args.weight = os.path.join(CACHE_DIR, f'{config.model_name}.pt')
        download_pretrained_weights(f'{config.model_name}', args.weight)
    version_name = config.model_name.split('v')[1]
    use_gpu = config.gpu  # False is use CPU

    net = YoloBackbone(
        use_gpu=use_gpu,
        version_name=version_name,
        weight=args.weight,
        num_classes=NUM_CLASSES,
        max_pre_nms=max_pre_nms,
        max_post_nms=max_post_nms)

    return net


class BaseBackbone(nn.Module):
    def __init__(self, **kwargs):
        super(BaseBackbone, self).__init__()
        pass

    def forward(self, batch):
        pass

    def detect(self, batch):
        pass


class BaseTimmModel(nn.Module):
    """Some Information about BaseTimmModel"""

    def __init__(
        self,
        num_classes,
        name="vit_base_patch16_224",
        from_pretrained=True,
        freeze_backbone=False,
    ):
        super().__init__()
        self.name = name
        self.model = timm.create_model(name, pretrained=from_pretrained)
        if name.find("nfnet") != -1:
            self.model.head.fc = nn.Linear(
                self.model.head.fc.in_features, num_classes)
        elif name.find("efficientnet") != -1:
            self.model.classifier = nn.Linear(
                self.model.classifier.in_features, num_classes
            )
        elif name.find("resnext") != -1:
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif name.find("vit") != -1:
            self.model.head = nn.Linear(
                self.model.head.in_features, num_classes)
        elif name.find("densenet") != -1:
            self.model.classifier = nn.Linear(
                self.model.classifier.in_features, num_classes
            )
        else:
            assert False, "Classifier block not included in TimmModel"

        self.model = nn.DataParallel(self.model)

    def forward(self, batch, device):
        inputs = batch["imgs"]
        inputs = inputs.to(device)
        outputs = self.model(inputs)
        return outputs


class YoloBackbone(BaseBackbone):
    def __init__(
            self,
            use_gpu,
            version_name,
            weight,
            num_classes=80,
            max_pre_nms=None,
            max_post_nms=None,
            **kwargs):

        super(YoloBackbone, self).__init__(**kwargs)

        if max_pre_nms is None:
            max_pre_nms = 30000
        self.max_pre_nms = max_pre_nms

        if max_post_nms is None:
            max_post_nms = 1000
        self.max_post_nms = max_post_nms

        version = version_name[0]
        if version == '4':
            version_mode = version_name.split('-')[1]
            self.name = f'yolov4-{version_mode}'
            self.model = Yolov4(
                cfg=f'./models/configs/yolov4-{version_mode}.yaml', ch=3, nc=num_classes
            )
        elif version == '5':
            version_mode = version_name[-1]
            self.name = f'yolov5{version_mode}'
            self.model = Model(
                cfg=f'./models/configs/yolov5{version_mode}.yaml', ch=3, nc=num_classes
            )
        if not use_gpu:
            map_loc = 'cpu'
        else:
            map_loc = None
        ckpt = torch.load(weight, map_location=map_loc)
        self.model.load_state_dict(
            ckpt['model'].state_dict(), strict=False)  # load state_dict

        self.loss_fn = YoloLoss(
            num_classes=num_classes,
            model=self.model)

        self.num_classes = num_classes

    def forward(self, batch, device):
        inputs = batch["imgs"]
        targets = batch['yolo_targets']

        inputs = inputs.to(device)
        targets = targets.to(device)

        if self.model.training:
            outputs = self.model(inputs)
        else:
            _, outputs = self.model(inputs)

        loss, loss_items = self.loss_fn(outputs, targets)

        ret_loss_dict = {
            'T': loss,
            'IOU': loss_items[0],
            'OBJ': loss_items[1],
            'CLS': loss_items[2],
        }
        return ret_loss_dict

    def detect(self, batch, device):
        inputs = batch["imgs"]
        inputs = inputs.to(device)
        outputs, _ = self.model(inputs)
        outputs = non_max_suppression(
            outputs,
            conf_thres=0.001,
            iou_thres=0.8,
            max_nms=self.max_pre_nms,
            max_det=self.max_post_nms)  # [bs, max_det, 6]

        out = []
        for i, output in enumerate(outputs):
            # [x1,y1,x2,y2, score, label]
            if output is not None and len(output) != 0:
                output = output.detach().cpu().numpy()
                boxes = output[:, :4]
                boxes[:, [0, 2]] = boxes[:, [0, 2]]
                boxes[:, [1, 3]] = boxes[:, [1, 3]]

                # Convert labels to COCO format
                labels = output[:, -1] + 1
                scores = output[:, -2]

            else:
                boxes = []
                labels = []
                scores = []
            if len(boxes) > 0:
                out.append({
                    'bboxes': boxes,
                    'classes': labels,
                    'scores': scores,
                })
            else:
                out.append({
                    'bboxes': np.array(()),
                    'classes': np.array(()),
                    'scores': np.array(()),
                })

        return out


def freeze_bn(model):
    def set_bn_eval(m):
        classname = m.__class__.__name__
        if "BatchNorm2d" in classname:
            m.affine = False
            m.weight.requires_grad = False
            m.bias.requires_grad = False
            m.eval()
    model.apply(set_bn_eval)

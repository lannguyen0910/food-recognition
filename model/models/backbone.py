# Author: Zylo117
import os
import torch
import torchvision
import numpy as np
from torch import nn
from .yolo import YoloLoss, Yolov4, non_max_suppression, Yolov5
from model.utils.utils import download_pretrained_weights

CACHE_DIR='./.cache'

def get_model(args, config, num_classes):
    
    NUM_CLASSES = num_classes
    max_post_nms = config.max_post_nms if config.max_post_nms > 0 else None
    max_pre_nms = config.max_pre_nms if config.max_pre_nms > 0 else None
    load_weights = True
    
    net = None

    version_name = config.model_name.split('v')[1]
    net = YoloBackbone(
        version_name=version_name,
        load_weights=load_weights, 
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

class YoloBackbone(BaseBackbone):
    def __init__(
        self,
        version_name='5s',
        num_classes=80, 
        max_pre_nms=None,
        load_weights=True, 
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
        if version=='4':
            version_mode = version_name.split('-')[1]
            self.name = f'yolov4-{version_mode}'
            self.model = Yolov4(
                cfg=f'./model/models/yolo/configs/yolov4-{version_mode}.yaml', ch=3, nc=num_classes
            )
        elif version =='5':
            version_mode = version_name[-1]
            self.name = f'yolov5{version_mode}'
            self.model = Yolov5(
                cfg=f'./model/models/yolo/configs/yolov5{version_mode}.yaml', ch=3, nc=num_classes
            )
        

        if load_weights:
            tmp_path = os.path.join(CACHE_DIR, f'yolo{version_name}.pth')
            download_pretrained_weights(f'yolov{version_name}', tmp_path)
            ckpt = torch.load(tmp_path, map_location='cpu')  # load checkpoint
            try:
                ret = self.model.load_state_dict(ckpt, strict=False) 
            except:
                pass
            print("Loaded pretrained model")

        self.model = nn.DataParallel(self.model).cuda()
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
            _ , outputs = self.model(inputs)

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
            conf_thres=0.0001, 
            iou_thres=0.8, 
            max_nms=self.max_pre_nms,
            max_det=self.max_post_nms) #[bs, max_det, 6]
    
        out = []
        for i, output in enumerate(outputs):
            # [x1,y1,x2,y2, score, label]
            if output is not None and len(output) != 0:
                output = output.detach().cpu().numpy()
                boxes = output[:, :4]
                boxes[:,[0,2]] = boxes[:,[0,2]] 
                boxes[:,[1,3]] = boxes[:,[1,3]] 

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




    

r"""
Usage:
    $ python path/to/export.py --weights yolov5s.pt --include torchscript
"""

from model.models.yolo.utils.torch_utils import select_device
from model.models.yolo.utils.general import (LOGGER, check_img_size, check_requirements, colorstr, file_size, print_args,
                                             url2file)
from model.models.yolo.utils.activations import SiLU
from model.models.yolo.yolo import Detect
from model.models.yolo.common import Conv
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile
import os
from model.configs.configs import Config
from model.models.detector import Detector
from model.utils.getter import *
from model.trainer.checkpoint import get_config, get_class_names

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = []
        for module in self:
            y.append(module(x, augment, profile, visualize)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, map_location=None, inplace=True, fuse=True):
    from model.models.yolo.yolo import Detect, Yolov5

    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        # ckpt = torch.load(
        #     w, map_location=map_location)  # load

        ignore_keys = [
            'min_iou_val',
            'min_conf_val',
            'tta',
            'gpu_devices',
            'tta_ensemble_mode',
            'tta_conf_threshold',
            'tta_iou_threshold',
        ]
        config = get_config(w, ignore_keys)

        if config is None:
            print("Config not found. Load configs from configs/configs.yaml")
            config = Config(os.path.join('configs', 'configs.yaml'))

        args = ''
        class_names, num_classes = get_class_names(w)

        net = get_model(
            args, config,
            num_classes=num_classes)

        ckpt = Detector(model=net, device='cpu')

        if fuse:
            # FP32 model
            model.append(
                ckpt.model.model._modules['module'].float().fuse().eval())
        else:
            # without layer fuse
            model.append(ckpt.model.model._modules['module'].float().eval())

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Yolov5]:
            m.inplace = inplace  # pytorch 1.7.0 compatibility
            if type(m) is Detect:
                if not isinstance(m.anchor_grid, list):  # new Detect Layer compatibility
                    delattr(m, 'anchor_grid')
                    setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print(f'Ensemble created with {weights}\n')
        for k in ['names']:
            setattr(model, k, getattr(model[-1], k))
        model.stride = model[torch.argmax(torch.tensor(
            [m.stride.max() for m in model])).int()].stride  # max stride
        return model  # return ensemble


def export_torchscript(model, im, file, optimize, prefix=colorstr('TorchScript:')):
    # YOLOv5 TorchScript model export
    try:
        print(f'\n{prefix} starting export with torch {torch.__version__}...')
        f = file.with_suffix('.torchscript.pt')
        fl = file.with_suffix('.torchscript.ptl')

        ts = torch.jit.trace(model, im, strict=False)
        (optimize_for_mobile(ts) if optimize else ts).save(f)
        (optimize_for_mobile(ts) if optimize else ts)._save_for_lite_interpreter(str(fl))

        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        print(f'{prefix} export success, saved as {fl} ({file_size(fl):.1f} MB)')
    except Exception as e:
        print(f'{prefix} export failure: {e}')


def export_onnx(model, im, file, opset, train, dynamic, simplify, prefix=colorstr('ONNX:')):
    # YOLOv5 ONNX export
    try:
        check_requirements(('onnx',))
        import onnx

        LOGGER.info(
            f'\n{prefix} starting export with onnx {onnx.__version__}...')
        f = file.with_suffix('.onnx')

        torch.onnx.export(model, im, f, verbose=False, opset_version=opset,
                          training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=not train,
                          input_names=['images'],
                          output_names=['output'],
                          dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
                                        # shape(1,25200,85)
                                        'output': {0: 'batch', 1: 'anchors'}
                                        } if dynamic else None)

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        # LOGGER.info(onnx.helper.printable_graph(model_onnx.graph))  # print

        # Simplify
        if simplify:
            try:
                check_requirements(('onnx-simplifier',))
                import onnxsim

                LOGGER.info(
                    f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(
                    model_onnx,
                    dynamic_input_shape=dynamic,
                    input_shapes={'images': list(im.shape)} if dynamic else None)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
            except Exception as e:
                LOGGER.info(f'{prefix} simplifier failure: {e}')
        LOGGER.info(
            f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        LOGGER.info(
            f"{prefix} run --dynamic ONNX model inference with: 'python detect.py --weights {f}'")
    except Exception as e:
        LOGGER.info(f'{prefix} export failure: {e}')


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # weights path
        imgsz=(320, 320),  # image (height, width)
        batch_size=1,  # batch size
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        include=('torchscript'),  # include formats
        half=False,  # FP16 half-precision export
        inplace=False,  # set YOLOv5 Detect() inplace=True
        train=False,  # model.train() mode
        optimize=True,  # TorchScript: optimize for mobile
        simplify=False,  # ONNX: simplify model
        opset=12,  # ONNX: opset version
        dynamic=False,  # ONNX/TF: dynamic axes
        ):
    t = time.time()
    include = [x.lower() for x in include]
    file = Path(url2file(weights) if str(weights).startswith(
        ('http:/', 'https:/')) else weights)

    # Checks
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand

    # Load PyTorch model
    device = select_device(device)
    assert not (
        device.type == 'cpu' and half), '--half only compatible with GPU export, i.e. use --device 0'
    model = attempt_load(weights, map_location=device,
                         inplace=True, fuse=False)  # load FP32 model

    # Input
    gs = int(max(model.stride))  # grid size (max stride)
    # verify img_size are gs-multiples
    imgsz = [check_img_size(x, gs) for x in imgsz]
    print('Image size: ', imgsz)
    # image size(1,3,320,192) BCHW iDetection
    print('Device: ', device)
    im = torch.zeros(batch_size, 3, *imgsz).to(device)

    # Update model
    if half:
        im, model = im.half(), model.half()  # to FP16

    # training mode = no Detect() layer grid construction
    print('Model: ', model)
    model.train() if train else model.eval()
    for k, m in model.named_modules():
        if isinstance(m, Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        elif isinstance(m, Detect):
            m.inplace = inplace
            m.onnx_dynamic = dynamic

    LOGGER.info(
        f"\n{colorstr('PyTorch:')} starting from {file} ({file_size(file):.1f} MB)")

    # Exports
    if 'torchscript' in include:
        model.model[-1].export = True  # set Detect() layer export=True
        _ = model(im)  # dry run

        export_torchscript(model, im, file, optimize)

        # traced_script_module = torch.jit.trace(model, im)
        # f = weights.replace('.pth', '.torchscript.pt')  # onnx filename

        # traced_script_module.save(f)
    if 'onnx' in include:  # OpenVINO requires ONNX
        export_onnx(model, im, file, opset, train, dynamic, simplify)

    # Finish
    LOGGER.info(f'\nExport complete ({time.time() - t:.2f}s)'
                f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
                f'\nVisualize with https://netron.app')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default=ROOT / 'yolov5s.pth', help='model.pth path(s)')
    parser.add_argument('--include', nargs='+',
                        default=['torchscript'],
                        help='available formats are (torchscript)')
    parser.add_argument('--opset', type=int, default=12,
                        help='ONNX: opset version')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+',
                        type=int, default=[320, 320], help='image (h, w)')
    parser.add_argument('--dynamic', action='store_true',
                        help='ONNX/TF: dynamic axes')
    parser.add_argument('--simplify', action='store_true',
                        help='ONNX: simplify model')

    opt = parser.parse_args()
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    for opt.weights in (opt.weights if isinstance(opt.weights, list) else [opt.weights]):
        run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

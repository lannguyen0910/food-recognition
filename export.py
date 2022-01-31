# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Export a YOLOv5 PyTorch model to other formats. TensorFlow exports authored by https://github.com/zldrobit
Format                      | `export.py --include`         | Model
---                         | ---                           | ---
PyTorch                     | -                             | yolov5s.pt
TorchScript                 | `torchscript`                 | yolov5s.torchscript
ONNX                        | `onnx`                        | yolov5s.onnx

Usage:
    $ python path/to/export.py --weights yolov5s.pt --include torchscript onnx openvino engine coreml tflite ...

"""

from models.utils.torch_utils import select_device
from models.utils.general import (LOGGER, check_img_size, check_requirements, colorstr,
                                  file_size, print_args, url2file)
from models.utils.activations import SiLU
from models.yolo import Detect
from models.experimental import attempt_load
from models.common import Conv
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def export_torchscript(model, im, file, optimize, prefix=colorstr('TorchScript:')):
    # YOLOv5 TorchScript model export
    try:
        LOGGER.info(
            f'\n{prefix} starting export with torch {torch.__version__}...')
        f = file.with_suffix('.torchscript')

        ts = torch.jit.trace(model, im, strict=False)
        d = {"shape": im.shape, "stride": int(
            max(model.stride)), "names": model.names}
        extra_files = {'config.txt': json.dumps(d)}  # torch._C.ExtraFilesMap()
        if optimize:  # https://pytorch.org/tutorials/recipes/mobile_interpreter.html
            optimize_for_mobile(ts)._save_for_lite_interpreter(
                str(f), _extra_files=extra_files)
        else:
            ts.save(str(f), _extra_files=extra_files)

        LOGGER.info(
            f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'{prefix} export failure: {e}')


def export_onnx(model, im, file, opset, train, dynamic, simplify, prefix=colorstr('ONNX:')):
    # YOLOv5 ONNX export
    try:
        check_requirements(('onnx',))
        import onnx

        LOGGER.info(
            f'\n{prefix} starting export with onnx {onnx.__version__}...')
        f = file.with_suffix('.onnx')
        print('f: ', f)

        torch.onnx.export(model, im, f, verbose=False, opset_version=opset,
                          input_names=['images'],
                          output_names=['output'],
                          dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
                                        # shape(1,25200,85)
                                        'output': {0: 'batch', 1: 'anchors'}
                                        } if dynamic else None)
        print('After export!')
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
        return f
    except Exception as e:
        LOGGER.info(f'{prefix} export failure: {e}')


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # weights path
        imgsz=(640, 640),  # image (height, width)
        batch_size=1,  # batch size
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        include=('torchscript', 'onnx'),  # include formats
        half=False,  # FP16 half-precision export
        inplace=False,  # set YOLOv5 Detect() inplace=True
        train=False,  # model.train() mode
        optimize=False,  # TorchScript: optimize for mobile
        dynamic=True,  # ONNX/TF: dynamic axes
        simplify=False,  # ONNX: simplify model
        opset=12,  # ONNX: opset version
        ):
    t = time.time()
    include = [x.lower() for x in include]
    file = Path(url2file(weights) if str(weights).startswith(
        ('http:/', 'https:/')) else weights)

    # Checks
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    # OpenVINO requires opset <= 12
    opset = 12 if ('openvino' in include) else opset

    # Load PyTorch model
    device = select_device(device)
    assert not (
        device.type == 'cpu' and half), '--half only compatible with GPU export, i.e. use --device 0'
    model = attempt_load(weights, map_location=device,
                         inplace=True, fuse=True)  # load FP32 model

    # Input
    gs = int(max(model.stride))  # grid size (max stride)
    # verify img_size are gs-multiples
    imgsz = [check_img_size(x, gs) for x in imgsz]
    # image size(1,3,320,192) BCHW iDetection
    im = torch.zeros(batch_size, 3, *imgsz).to(device)

    # Update model
    if half:
        im, model = im.half(), model.half()  # to FP16
    # training mode = no Detect() layer grid construction
    model.train() if train else model.eval()
    for k, m in model.named_modules():
        if isinstance(m, Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        elif isinstance(m, Detect):
            m.inplace = inplace
            m.onnx_dynamic = dynamic
            if hasattr(m, 'forward_export'):
                # assign custom forward (optional)
                m.forward = m.forward_export

    for _ in range(2):
        y = model(im)  # dry runs
    LOGGER.info(
        f"\n{colorstr('PyTorch:')} starting from {file} ({file_size(file):.1f} MB)")

    # Exports
    f = [''] * 10  # exported filenames
    if 'torchscript' in include:
        f[0] = export_torchscript(model, im, file, optimize)

    if ('onnx' in include):  # OpenVINO requires ONNX
        f[2] = export_onnx(model, im, file, opset, train, dynamic, simplify)

    # Finish
    f = [str(x) for x in f if x]  # filter out '' and None
    print('AFter f: ', f)
    LOGGER.info(f'\nExport complete ({time.time() - t:.2f}s)'
                f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
                f"\nVisualize with https://netron.app"
                f"\nDetect with `python detect.py --weights {f[-1]}`"
                f" or `model = torch.hub.load('ultralytics/yolov5', 'custom', '{f[-1]}')"
                f"\nValidate with `python val.py --weights {f[-1]}`")
    return f  # return list of exported files/dirs


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+',
                        type=int, default=[640, 640], help='image (h, w)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='cpu',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true',
                        help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true',
                        help='set YOLOv5 Detect() inplace=True')
    parser.add_argument('--train', action='store_true',
                        help='model.train() mode')
    parser.add_argument('--optimize', action='store_true',
                        help='TorchScript: optimize for mobile')
    parser.add_argument('--dynamic', action='store_true',
                        help='ONNX/TF: dynamic axes')
    parser.add_argument('--simplify', action='store_true',
                        help='ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=12,
                        help='ONNX: opset version')
    parser.add_argument('--include', nargs='+',
                        default=['torchscript', 'onnx'],
                        help='torchscript, onnx, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs')
    opt = parser.parse_args()
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    for opt.weights in (opt.weights if isinstance(opt.weights, list) else [opt.weights]):
        run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

from utilities.utils.getter import *
import argparse

parser = argparse.ArgumentParser(description='Perfom Objet Detection')
parser.add_argument('--weight', type=str, default=None,
                    help='version of EfficentDet')
parser.add_argument('--input_path', type=str,
                    help='path to an image to inference')
parser.add_argument('--output_path', type=str,
                    help='path to save inferenced image')
parser.add_argument('--gpus', type=str, default=False,
                    help='path to save inferenced image')
parser.add_argument('--min_conf', type=float, default=0.15,
                    help='minimum confidence for an object to be detect')
parser.add_argument('--min_iou', type=float, default=0.5,
                    help='minimum iou threshold for non max suppression')
parser.add_argument('--tta', action='store_true',
                    help='whether to use test time augmentation')
parser.add_argument('--tta_ensemble_mode', type=str,
                    default='wbf', help='tta ensemble mode')
parser.add_argument('--tta_conf_threshold', type=float,
                    default=0.01, help='tta confidence score threshold')
parser.add_argument('--tta_iou_threshold', type=float,
                    default=0.9, help='tta iou threshold')

CACHE_DIR = './.cache'

# Global model, only changes when model name changes
DETECTOR = None


class Testset():
    def __init__(self, config, input_path, transforms=None):
        self.input_path = input_path  # path to image folder or a single image
        self.transforms = transforms
        self.image_size = config.image_size
        self.load_images()

    def get_batch_size(self):
        num_samples = len(self.all_image_paths)

        # Temporary
        return 1

    def load_images(self):
        self.all_image_paths = []
        if os.path.isdir(self.input_path):  # path to image folder
            paths = sorted(os.listdir(self.input_path))
            for path in paths:
                self.all_image_paths.append(
                    os.path.join(self.input_path, path))
        elif os.path.isfile(self.input_path):  # path to single image
            self.all_image_paths.append(self.input_path)

    def __getitem__(self, idx):
        image_path = self.all_image_paths[idx]
        image_name = os.path.basename(image_path)
        img = cv2.imread(image_path)
        image_w, image_h = self.image_size
        ori_height, ori_width, c = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0
        ori_img = img.copy()
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        return {
            'img': img,
            'img_name': image_name,
            'ori_img': ori_img,
            'image_ori_w': ori_width,
            'image_ori_h': ori_height,
            'image_w': image_w,
            'image_h': image_h,
        }

    def collate_fn(self, batch):
        imgs = torch.stack([s['img'] for s in batch])
        ori_imgs = [s['ori_img'] for s in batch]
        img_names = [s['img_name'] for s in batch]
        image_ori_ws = [s['image_ori_w'] for s in batch]
        image_ori_hs = [s['image_ori_h'] for s in batch]
        image_ws = [s['image_w'] for s in batch]
        image_hs = [s['image_h'] for s in batch]
        img_scales = torch.tensor([1.0]*len(batch), dtype=torch.float)
        img_sizes = torch.tensor(
            [imgs[0].shape[-2:]]*len(batch), dtype=torch.float)
        return {
            'imgs': imgs,
            'ori_imgs': ori_imgs,
            'img_names': img_names,
            'image_ori_ws': image_ori_ws,
            'image_ori_hs': image_ori_hs,
            'image_ws': image_ws,
            'image_hs': image_hs,
            'img_sizes': img_sizes,
            'img_scales': img_scales
        }

    def __len__(self):
        return len(self.all_image_paths)

    def __str__(self):
        return f"Number of found images: {len(self.all_image_paths)}"


def detect(args, config):
    global DETECTOR

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    test_transforms = A.Compose([
        get_resize_augmentation(
            config.image_size, keep_ratio=config.keep_ratio),
        A.Normalize(mean=MEAN, std=STD, max_pixel_value=1.0, p=1.0),
        ToTensorV2(p=1.0)
    ])

    if args.tta:
        args.tta = TTA(
            min_conf=args.tta_conf_threshold,
            min_iou=args.tta_iou_threshold,
            postprocess_mode=args.tta_ensemble_mode)
    else:
        args.tta = None

    testset = Testset(
        config,
        args.input_path,
        transforms=test_transforms)
    testloader = DataLoader(
        testset,
        batch_size=testset.get_batch_size(),
        num_workers=2,
        pin_memory=True,
        collate_fn=testset.collate_fn
    )

    class_names, num_classes = config.names, config.nc
    class_names.insert(0, 'Background')

    if DETECTOR is None or DETECTOR.model_name != config.model_name:
        net = get_model(args, config, num_classes=num_classes)
        DETECTOR = Detector(model=net, freeze=True, device=device)

        # Print info
        print(config)

    DETECTOR.eval()

    for param in DETECTOR.parameters():
        param.requires_grad = False

    result_dict = {
        'boxes': [],
        'labels': [],
        'scores': []
    }

    empty_imgs = 0
    with tqdm(total=len(testloader)) as pbar:
        with torch.no_grad():
            for idx, batch in enumerate(testloader):
                if args.tta is not None:
                    preds = args.tta.make_tta_predictions(DETECTOR, batch)
                else:
                    preds = DETECTOR.inference_step(batch)

                for idx, outputs in enumerate(preds):
                    img_w = batch['image_ws'][idx]
                    img_h = batch['image_hs'][idx]
                    img_ori_ws = batch['image_ori_ws'][idx]
                    img_ori_hs = batch['image_ori_hs'][idx]
                    outputs = postprocessing(
                        outputs,
                        current_img_size=[img_w, img_h],
                        ori_img_size=[img_ori_ws, img_ori_hs],
                        min_iou=args.min_iou,
                        min_conf=args.min_conf,
                        max_dets=config.max_post_nms,
                        keep_ratio=config.keep_ratio,
                        output_format='xywh',
                        mode=config.fusion_mode)

                    boxes = outputs['bboxes']
                    labels = outputs['classes']
                    scores = outputs['scores']

                    for (box, label, score) in zip(boxes, labels, scores):
                        result_dict['boxes'].append(box)
                        result_dict['labels'].append(label)
                        result_dict['scores'].append(score)

                    if len(boxes) == 0:
                        empty_imgs += 1
                        boxes = None

                pbar.update(1)
                pbar.set_description(f'Empty images: {empty_imgs}')
    return result_dict

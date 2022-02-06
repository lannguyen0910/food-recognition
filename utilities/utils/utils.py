import torch
import torch.nn as nn
import torchvision
import numpy as np
import math
import webcolors
import cv2
import gdown
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_

STANDARD_COLORS = [
    'LawnGreen', 'LightBlue', 'Crimson', 'Gold', 'Azure', 'BlanchedAlmond', 'Bisque',
    'Aquamarine', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'AliceBlue', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def from_colorname_to_bgr(color):
    rgb_color = webcolors.name_to_rgb(color)
    result = (rgb_color.blue, rgb_color.green, rgb_color.red)
    return result


def standard_to_bgr(list_color_name):
    standard = []
    for i in range(len(list_color_name) - 26):  # -26 used to match the len(obj_list)
        standard.append(from_colorname_to_bgr(list_color_name[i]))
    return standard


color_list = standard_to_bgr(STANDARD_COLORS)


def draw_boxes_v2(img_name, img, boxes, label_ids, scores, label_names=None, obj_list=None, figsize=(15, 15)):
    """
    Visualize an image with its bouding boxes
    rgn image + xywh box
    """
    def plot_one_box(img, box, key=None, value=None, color=None, line_thickness=None):
        tl = line_thickness or int(
            round(0.001 * max(img.shape[0:2])))  # line thickness

        coord = [box[0], box[1], box[0]+box[2], box[1]+box[3]]
        c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl*2)
        if key is not None and value is not None:
            header = f'{key}: {value}'
            tf = max(tl - 2, 1)  # font thickness
            s_size = cv2.getTextSize(
                f' {value}', 0, fontScale=float(tl) / 3, thickness=tf)[0]
            t_size = cv2.getTextSize(
                f'{key}:', 0, fontScale=float(tl) / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0] + s_size[0] + 15, c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1)  # filled
            cv2.putText(img, header, (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0],
                        thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)

    # boxes input is xywh
    boxes = boxes.astype(np.int)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for idx, (box, label_id, score) in enumerate(zip(boxes, label_ids, scores)):
        if label_names is not None:
            label = label_names[idx]
        if obj_list is not None:
            label = obj_list[label_id]

        plot_one_box(
            img_bgr,
            box,
            key=label,
            value='{:.0%}'.format(float(score)),
            color=color_list[int(label_id)])

    cv2.imwrite(img_name, img_bgr)


def draw_pred_gt_boxes(image_outname, img, boxes, labels, scores, image_name=None, figsize=(10, 10)):
    """
    Visualize an image with its bouding boxes
    """
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    # if image_name is not None:
    #     fig.suptitle(image_name)
    #     fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if isinstance(img, torch.Tensor):
        img = img.numpy().squeeze().transpose((1, 2, 0))
    # Display the image
    ax1.imshow(img)
    ax2.imshow(img)

    ax1.set_title('Prediction')
    ax2.set_title('Ground Truth')

    # Split prediction  and ground truth
    pred_boxes, pred_labels, pred_scores = boxes[0], labels[0], scores
    gt_boxes, gt_labels = boxes[1], labels[1]

    # Plot prediction boxes
    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        label = int(label)
        color = STANDARD_COLORS[label]
        x, y, w, h = box
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=1.5, edgecolor=color, facecolor='none')
        score = np.round(score, 3)
        text = '{}: {}'.format(label, str(score))
        ax1.text(x, y-3, text, color=color, fontsize=15)
        # Add the patch to the Axes
        ax1.add_patch(rect)

    # Plot ground truth boxes
    for box, label in zip(gt_boxes, gt_labels):
        label = int(label)
        if label < 0:
            continue
        color = STANDARD_COLORS[label]
        x, y, w, h = box
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=1.5, edgecolor=color, facecolor='none')
        score = np.round(score, 3)
        text = '{}'.format(label)
        ax2.text(x, y-3, text, color=color, fontsize=15)
        # Add the patch to the Axes
        ax2.add_patch(rect)

    plt.axis('off')
    fig = ax1.get_figure()
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)

    fig = ax2.get_figure()
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)

    plt.savefig(image_outname, bbox_inches='tight')
    return fig

    # plt.close()


def write_to_video(img, boxes, labels, scores, imshow=True,  outvid=None, obj_list=None):

    def plot_one_box(img, box, key=None, value=None, color=None, line_thickness=None):
        tl = line_thickness or int(
            round(0.001 * max(img.shape[0:2])))  # line thickness

        coord = [box[0], box[1], box[0]+box[2], box[1]+box[3]]
        c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl*2)
        if key is not None and value is not None:
            header = f'{key} || {value}'
            tf = max(tl - 2, 1)  # font thickness
            s_size = cv2.getTextSize(
                f'| {value}', 0, fontScale=float(tl) / 3, thickness=tf)[0]
            t_size = cv2.getTextSize(
                f'{key} |', 0, fontScale=float(tl) / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0] + s_size[0] + 15, c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1)  # filled
            cv2.putText(img, header, (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0],
                        thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)

    # boxes input is xywh
    boxes = boxes.astype(np.int)

    for idx, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        plot_one_box(
            img,
            box,
            key=obj_list[int(label)],
            value='{:.0%}'.format(float(score)),
            color=color_list[int(label)])

    if imshow:
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 600, 600)
        cv2.imshow('img', img)
        cv2.waitKey(1)

    if outvid is not None:
        outvid.write(img)


def download_weights(id_or_url, cached=None, md5=None, quiet=False):
    if id_or_url.startswith('http'):
        url = id_or_url
    else:
        url = 'https://drive.google.com/uc?id={}'.format(id_or_url)

    return gdown.cached_download(url=url, path=cached, md5=md5, quiet=quiet)

# https://drive.google.com/file/d/1uIp1Kl1xWsVqiE2tDPChEnTo1ZOwC0Jv/view?usp=sharing
weight_url = {
    'yolov5s': "1rISMag8OCM5v99TYuavAobm3LkwjtAi9",
    "yolov5m": "1I649VGqkam_IcCCW8WUA965vPrW_pqDX",
    "yolov5l": "1sBciFcRav2ZE6jzhWnca9uegjQ4860om",
    "yolov5x": "1CRD6T9QtH9XEa-h985_Ho6jgLWu58zn0",
    "se_resnet": "1SjWV-tZ980n7t0K68Lfpu5j9sw5-E2Gs"
}


def download_pretrained_weights(name, cached=None):
    return download_weights(weight_url[name], cached)

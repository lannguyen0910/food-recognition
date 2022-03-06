import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .colors import color_list


def draw_mask(polygons, mask_img):
    ImageDraw.Draw(mask_img).polygon(polygons, outline=1, fill=1)
    return mask_img


def draw_polylines(image, polygons):
    image = cv2.polylines(
        image, [np.array(polygons, dtype=np.int)], True, (0, 0, 1), 2)
    return image


def draw_text(image, text, polygons, font, color, font_size):
    im = Image.fromarray(np.uint8(image*255))
    draw = ImageDraw.Draw(im)
    unicode_font = ImageFont.truetype(font, font_size)
    draw.text((polygons[0][0], polygons[0][1]+10),
              text, font=unicode_font, fill=color)
    return np.asarray(im)/255


def get_font_size(image, text, polygons, font_type):
    fontsize = 1  # starting font size

    polywidth = polygons[1][0] - polygons[0][0]
    imagewidth = image.shape[1]

    # portion of image width you want text width to be
    img_fraction = 1

    font = ImageFont.truetype(font_type, fontsize)

    idx = 100
    while font.getsize(text)[0] < img_fraction*polywidth:
        # iterate until the text size is just larger than the criteria
        fontsize += 1
        font = ImageFont.truetype(font_type, fontsize)
        idx -= 1

        if idx <= 0:
            break
    fontsize -= 1
    return fontsize


def reduce_opacity(image):
    # pre-multiplication
    a_channel = np.ones(image.shape, dtype=np.float)/3.0
    img = image*a_channel
    return img


def draw_text_cv2(
    img,
    text,
    uv_top_left,
    color=(255, 255, 255),
    fontScale=0.5,
    thickness=1,
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    outline_color=(0, 0, 0),
    line_spacing=1.5,
):
    """
    Draws multiline with an outline.
    """
    assert isinstance(text, str)

    uv_top_left = np.array(uv_top_left, dtype=float)
    assert uv_top_left.shape == (2,)

    for line in text.splitlines():
        (w, h), _ = cv2.getTextSize(
            text=line,
            fontFace=fontFace,
            fontScale=fontScale,
            thickness=thickness,
        )
        uv_bottom_left_i = uv_top_left + [0, h]
        org = tuple(uv_bottom_left_i.astype(int))

        if outline_color is not None:
            cv2.putText(
                img,
                text=line,
                org=org,
                fontFace=fontFace,
                fontScale=fontScale,
                color=outline_color,
                thickness=thickness * 3,
                lineType=cv2.LINE_AA,
            )
        cv2.putText(
            img,
            text=line,
            org=org,
            fontFace=fontFace,
            fontScale=fontScale,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        uv_top_left += [0, h * line_spacing]


def draw_bboxes_v2(savepath, img, boxes, label_ids, scores, label_names=None, obj_list=None):
    """
    Visualize an image with its bouding boxes
    rgb image + xywh box
    """
    def plot_one_box(img, box, key=None, value=None, color=None, line_thickness=None):
        tl = line_thickness or int(
            round(0.001 * max(img.shape[0:2])))  # line thickness

        coord = [box[0], box[1], box[0]+box[2], box[1]+box[3]]
        c1, c2 = (int(coord[0]), int(coord[1])
                  ), (int(coord[2]), int(coord[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl+1)

        if key is not None and value is not None:
            header = f'{key}: {value}'
            tf = max(tl - 2, 2)  # font thickness
            s_size = cv2.getTextSize(
                f' {value}', 0, fontScale=float(tl) / 3, thickness=tf)[0]
            t_size = cv2.getTextSize(
                f'{key}:', 0, fontScale=float(tl) / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0] + s_size[0] + 15, c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1)  # filled
            cv2.putText(img, header, (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0],
                        thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)

    # boxes input is xywh
    boxes = np.array(boxes, np.int)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for idx, (box, label_id, score) in enumerate(zip(boxes, label_ids, scores)):
        if label_names is not None:
            label = label_names[idx]
        if obj_list is not None:
            label = obj_list[label_id]
        
        new_color = tuple(i*255.0 for i in color_list[int(label_id)])

        plot_one_box(
            img_bgr,
            box,
            key=label,
            value='{:.0%}'.format(float(score)),
            color=new_color,
            line_thickness=2)

    cv2.imwrite(savepath, img_bgr)

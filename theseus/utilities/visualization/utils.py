import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def draw_mask(polygons, mask_img):
    ImageDraw.Draw(mask_img).polygon(polygons, outline=1, fill=1)
    return mask_img

def draw_polylines(image, polygons):
    image = cv2.polylines(image, [np.array(polygons, dtype=np.int)], True, (0,0,1), 2)
    return image

def draw_text(image, text, polygons, font, color, font_size):
    im  =  Image.fromarray(np.uint8(image*255))
    draw  =  ImageDraw.Draw ( im )
    unicode_font = ImageFont.truetype(font, font_size)
    draw.text((polygons[0][0],polygons[0][1]+10), text, font=unicode_font, fill=color )
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

        if idx <=0:
            break
    fontsize -= 1
    return fontsize

def reduce_opacity(image):
    #pre-multiplication
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
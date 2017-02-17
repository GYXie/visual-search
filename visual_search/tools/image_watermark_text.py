# -*- coding=utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.misc import imread
from scipy import misc
from scipy.misc import imread
import matplotlib.cbook as cbook
from PIL import Image, ImageDraw, ImageFont

args = None


# 参考: http://www.thecodingcouple.com/watermark-images-python-pillow-pil/
def main():
    image = Image.open(args.input_path)
    width, height = image.size
    draw = ImageDraw.Draw(image)
    text = "Deep Learning"

    font = ImageFont.truetype('/Library/Fonts/Arial.ttf', 30)
    text_width, text_height = draw.textsize(text, font)

    margin = 10
    x = width - text_width - margin
    y = height - text_height - margin
    draw.text((x, y), text, font=font, fill=(255, 0, 0))
    image.save(args.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path',
        type=str,
        default='./g.jpg',
        help='Image file path.'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='g_watermark_text.jpg',
        help='Output file path.'
    )
    args = parser.parse_args()
    print(args)
    main()

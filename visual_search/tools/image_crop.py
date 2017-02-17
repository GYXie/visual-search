# -*- coding=utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.misc import imread
from scipy import misc
from PIL import Image, ImageDraw, ImageFont

args = None

def main():
    image = Image.open(args.input_path)
    width, height = image.size
    box = (width * 0.1, height * 0.1, width * 0.9, height * 0.9)
    crop = image.crop(box)
    print(crop.size)
    image = Image.new('RGBA', crop.size)
    box = (0, 0, crop.size[0], crop.size[1])
    image.paste(crop, box)
    image.save(args.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path',
        type=str,
        default='g.jpg',
        help='Image file path.'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='g_crop.jpg',
        help='Output file path.'
    )
    args = parser.parse_args()
    print(args)
    main()

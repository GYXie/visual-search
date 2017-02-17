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
from PIL import Image, ImageOps

args = None

def main():
    image = Image.open(args.input_path)
    image = ImageOps.mirror(image)
    # data = np.asarray(image)
    # print(type(data))
    # print(type(data[0,0,0]))
    # print(data.shape)
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
        default='g_mirror.jpg',
        help='Output file path.'
    )
    args = parser.parse_args()
    print(args)
    main()

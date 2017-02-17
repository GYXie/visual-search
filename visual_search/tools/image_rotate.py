# -*- coding=utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.misc import imread
from scipy import misc

args = None

# https://github.com/tiagopereira/python_tips/wiki/Scipy:-image-rotation
def main():
    img = imread(args.input_path)
    img = ndimage.rotate(img, args.angle, mode=args.mode)
    misc.imsave(args.output_path, img)



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
        default='g_rotate.jpg',
        help='Output file path.'
    )
    parser.add_argument(
        '--angle',
        type=float,
        default='15',
        help='Rotate angle.'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='nearest',
        help='Rotate mode.'
    )
    args = parser.parse_args()
    print(args)
    main()

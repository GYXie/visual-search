# -*- coding=utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import argparse
import os.path
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread

args = None


def main():
    args.input_data_dir = os.path.abspath(args.input_data_dir)
    if not os.path.exists(args.output_data_dir):
        os.mkdir(args.output_data_dir)
    for dir_path, dir_names, file_names in os.walk(args.input_data_dir):
        if len(file_names) > 0:
            print(dir_path)
            rows = int(math.ceil(len(file_names) / 6.0))
            print(rows)
            fig, axes = plt.subplots(4, 12, subplot_kw={'xticks': [], 'yticks': []})
            fig.subplots_adjust(hspace=0.01, wspace=0.01)
            for ax, file_name in zip(axes.flat, file_names):
                print(file_name)
                img = imread(dir_path + '/' + file_name)
                ax.imshow(img)
                # ax.set_title(os.path.splitext(file_name)[0].replace('.227x227', ''))
            plt.savefig(args.output_data_dir + dir_path.replace(args.input_data_dir, '') + '.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_data_dir',
        type=str,
        default='/Volumes/Transcend/dataset/Caltech256/256_ObjectCategories_resize',
        help='Directory to put the input data.'
    )
    parser.add_argument(
        '--output_data_dir',
        type=str,
        default='/Volumes/Transcend/dataset/Caltech256/256_ObjectCategories_plot',
        help='Directory to put the output file.'
    )
    args = parser.parse_args()
    print(args)
    main()
    # methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
    #            'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
    #            'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
    #
    # grid = np.random.rand(4, 4)
    #
    # fig, axes = plt.subplots(3, 6, figsize=(12, 6),
    #                          subplot_kw={'xticks': [], 'yticks': []})
    #
    # fig.subplots_adjust(hspace=0.3, wspace=0.05)
    #
    # for ax, interp_method in zip(axes.flat, methods):
    #     ax.imshow(grid, interpolation=interp_method)
    #     ax.set_title(interp_method)
    #
    # plt.show()

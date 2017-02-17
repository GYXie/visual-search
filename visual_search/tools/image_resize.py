# -*- coding=utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import argparse
import os.path
from scipy import misc
from scipy.misc import imread

args = None


def main():
    print(args)
    for dir_path, dir_names, file_names in os.walk(args.input_data_dir):
        # dir_path is a string, the path to the directory
        # dir_names is a list of the names of the subdirectories in dir_path (excluding '.' and '..')
        # file_names is a list of the names of the non-directory files in dir_path
        dir_absolute_path = args.output_data_dir + dir_path.replace(args.input_data_dir, '')
        if not os.path.exists(dir_absolute_path):
            os.mkdir(dir_absolute_path)
        for file_name in file_names:
            # Split the pathname path into a pair (root, ext) such that root + ext == path, and ext is empty or begins
            # with a period and contains at most one period.
            (root, ext) = os.path.splitext(file_name)
            new_file_name = '%s/%s.%dx%d%s' % (
                dir_absolute_path, root, args.width, args.height, ext)
            print(new_file_name)
            if not os.path.exists(new_file_name):
                img = imread(dir_path + '/' + file_name)
                # type(img) = ndarray, https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html
                (width, height) = img.shape[0:2]
                if width > height:
                    size = (args.width, height * args.width / width)
                else:
                    size = (width * args.height / height, args.height)
                new_img = misc.imresize(img, size)
                misc.imsave(new_file_name, new_img)


if __name__ == '__main__':
    """Resize the images in [input_dir] and save them in [output_dir]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_data_dir',
        type=str,
        help='Directory to put the input data.'
    )
    parser.add_argument(
        '--output_data_dir',
        type=str,
        help='Directory to put the output data.'
    )

    parser.add_argument(
        '--width',
        type=int,
        default=227,
        help='Target image width.'
    )

    parser.add_argument(
        '--height',
        type=int,
        default=227,
        help='Target image height.'
    )
    args = parser.parse_args()
    main()

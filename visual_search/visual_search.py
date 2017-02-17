# coding=utf-8
from __future__ import absolute_import
from __future__ import print_function

import argparse
import time
import urllib
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, ImageDraw, ImageFont
from numpy import *
from scipy import ndimage
from scipy.misc import imread
from scipy.misc import imresize
from scipy.spatial import distance
from sklearn import preprocessing

args = None
train_x = zeros((1, 227, 227, 3)).astype(float32)
train_y = zeros((1, 1000))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]

net_data = load("bvlc_alexnet.npy").item()


def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i % group == 0
    assert c_o % group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

    if group == 1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])


def initModel():
    x = tf.placeholder(tf.float32, (None,) + xdim)
    # conv1
    # conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11;
    k_w = 11;
    c_o = 96;
    s_h = 4;
    s_w = 4
    conv1W = tf.Variable(net_data["conv1"][0])
    conv1b = tf.Variable(net_data["conv1"][1])
    conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)

    # lrn1
    # lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2;
    alpha = 2e-05;
    beta = 0.75;
    bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    # maxpool1
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3;
    k_w = 3;
    s_h = 2;
    s_w = 2;
    padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    # conv2
    # conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5;
    k_w = 5;
    c_o = 256;
    s_h = 1;
    s_w = 1;
    group = 2
    conv2W = tf.Variable(net_data["conv2"][0])
    conv2b = tf.Variable(net_data["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)

    # lrn2
    # lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2;
    alpha = 2e-05;
    beta = 0.75;
    bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    # maxpool2
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
    k_h = 3;
    k_w = 3;
    s_h = 2;
    s_w = 2;
    padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    # conv3
    # conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3;
    k_w = 3;
    c_o = 384;
    s_h = 1;
    s_w = 1;
    group = 1
    conv3W = tf.Variable(net_data["conv3"][0])
    conv3b = tf.Variable(net_data["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)

    # conv4
    # conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3;
    k_w = 3;
    c_o = 384;
    s_h = 1;
    s_w = 1;
    group = 2
    conv4W = tf.Variable(net_data["conv4"][0])
    conv4b = tf.Variable(net_data["conv4"][1])
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)

    # conv5
    # conv(3, 3, 256, 1, 1, group=2, name='conv5')
    k_h = 3;
    k_w = 3;
    c_o = 256;
    s_h = 1;
    s_w = 1;
    group = 2
    conv5W = tf.Variable(net_data["conv5"][0])
    conv5b = tf.Variable(net_data["conv5"][1])
    conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv5 = tf.nn.relu(conv5_in)

    # maxpool5
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
    k_h = 3;
    k_w = 3;
    s_h = 2;
    s_w = 2;
    padding = 'VALID'
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    # fc6
    # fc(4096, name='fc6')
    fc6W = tf.Variable(net_data["fc6"][0])
    fc6b = tf.Variable(net_data["fc6"][1])
    fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

    # fc7
    # fc(4096, name='fc7')
    fc7W = tf.Variable(net_data["fc7"][0])
    fc7b = tf.Variable(net_data["fc7"][1])
    fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

    # fc8
    # fc(1000, relu=False, name='fc8')
    fc8W = tf.Variable(net_data["fc8"][0])
    fc8b = tf.Variable(net_data["fc8"][1])
    fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

    # prob
    # softmax(name='prob'))
    prob = tf.nn.softmax(fc8)

    return (x, fc6)


def load_image(img_file_path):
    img = imread(img_file_path)
    img = (imresize(img, (227, 227))[:, :, :3]).astype(float32)
    img = img - mean(img)

    return img


def extract_feature(imgs):
    x, fc6 = initModel()
    # init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    return sess.run(fc6, feed_dict={x: imgs})


def load_image_names():
    img_names = []
    with open('img_names.txt') as f:
        for img_name in f.readlines():
            img_name = img_name.strip('\n')
            img_names.append(img_name)
    return img_names


def download_image(url):
    tmp_img_path = 'img/tmp.jpg'
    urllib.urlretrieve(url, tmp_img_path)
    return tmp_img_path


def crop(np_img):
    image = Image.fromarray(np_img)
    width, height = image.size
    box = (width * 0.2, height * 0.2, width * 0.8, height * 0.8)
    crop = image.crop(box)
    image = Image.new('RGBA', crop.size)
    box = (0, 0, crop.size[0], crop.size[1])
    image.paste(crop, box)
    return np.asarray(image)


def rotate(np_img):
    img = ndimage.rotate(np_img, 10, mode='nearest')
    return img


def watermark(np_img):
    image = Image.fromarray(np_img)
    width, height = image.size
    draw = ImageDraw.Draw(image)
    text = "Deep Learning"

    font = ImageFont.truetype('/Library/Fonts/Arial.ttf', 30)
    text_width, text_height = draw.textsize(text, font)

    margin = 10
    x = width - text_width - margin
    y = height / 2 - text_height - margin
    draw.text((x, y), text, font=font, fill=(255, 0, 0))
    return np.asarray(image)


def mirror(np_img):
    image = Image.fromarray(np_img)
    image = ImageOps.mirror(image)
    data = np.asarray(image)
    return data


def image_normalize(imgs):
    new_images = []
    for img in imgs:
        img = (imresize(img, (227, 227))[:, :, :3]).astype(float32)
        img -= mean(img)
        new_images.append(img)
    return new_images


def main():
    t = time.time()
    img = imread(args.img_file_path)
    imgs = [img, watermark(img), rotate(img), crop(img), mirror(img)]
    imgs_norm = image_normalize(imgs)
    dataset_features = np.load('fc6.npy')

    query_start = time.time()
    query_features = extract_feature(imgs_norm)
    binarizer = preprocessing.Binarizer().fit(query_features)
    query_features = binarizer.transform(query_features)
    print(dataset_features)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist
    cosine = distance.cdist(dataset_features, query_features, 'cosine')
    print(cosine.shape)
    dis = cosine
    inds_all = argsort(dis, axis=0)  # 按列排序 https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
    print('query cost: %f, dataset: %d, query: %d' % (time.time() - query_start, len(dataset_features), len(imgs)))
    img_names = load_image_names()
    fig, axes = plt.subplots(5, 11, figsize=(22, 10), subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(hspace=0.15, wspace=0.01, left=.02, right=.98, top=.92, bottom=.08)
    titles = ['original', 'watermark', 'rotate', 'crop', 'mirror']
    for i in range(len(imgs)):
        topK = []
        inds = inds_all[:, i]
        # print(inds)
        for k in range(10):
            topK.append(img_names[inds[k]])
            print(inds[k], dis[inds[k], i], img_names[inds[k]])

        original = axes[i, 0]
        original.set_title(titles[i])
        img = imgs[i]
        original.imshow(img)
        for j in range(10):
            ax = axes[i, j + 1]
            img = imread(topK[j])
            ax.imshow(img)
            title = '%d : %f' % (j + 1, dis[inds[j], i])
            ax.set_title(title)

    savePath = args.img_file_path + '_search_result.jpg'
    plt.savefig(savePath)
    print(time.time() - t)
    # os.system('open -a Preview.app -F ' + savePath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--img_file_path',
        type=str,
        default='041_0010.jpg',
        help='Image file path.'
    )
    parser.add_argument(
        '--img_url',
        type=str,
        default='',
        help='Image Url.'
    )
    args = parser.parse_args()
    if not args.img_url == '':
        args.img_file_path = download_image(args.url)
    main()

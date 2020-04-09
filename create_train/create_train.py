from __future__ import division
import numpy as np
import sys
sys.path.append("./mingqingscript")
import scipy.ndimage.interpolation
# import scipy.signal
import os
import sys
import matplotlib.pyplot as plt
from random import uniform
import h5py
from PIL import Image


def array2PIL(arr, size):
    mode = 'RGBA'
    arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = np.c_[arr, 255 * numpy.ones((len(arr), 1), numpy.uint8)]
    return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)




def split_data(split='train'):
    nyu_depth = h5py.File('nyu_depth_v2_labeled.mat', 'r')
    if(split=='train'):
        directory='facades/train'
        start=0
        end=1000
    else:
        directory='facades/val'
        start=1001
        end=1100

    if not os.path.exists(directory):
        os.makedirs(directory)
    image = nyu_depth['images']
    depth = nyu_depth['depths']
    img_size = 224
    total_num = 0


    for index in range(start,end):
        index = index
        gt_image = (image[index, :, :, :]).astype(float)
        gt_image = np.swapaxes(gt_image, 0, 2)
        gt_image = scipy.misc.imresize(gt_image, [img_size, img_size]).astype(float)
        gt_image = gt_image / 255
        gt_depth = depth[index, :, :]
        maxhazy = gt_depth.max()
        minhazy = gt_depth.min()
        gt_depth = (gt_depth) / (maxhazy)
        gt_depth = np.swapaxes(gt_depth, 0, 1)
        scale1 = (gt_depth.shape[0]) / img_size
        scale2 = (gt_depth.shape[1]) / img_size
        gt_depth = scipy.ndimage.zoom(gt_depth, (1 / scale1, 1 / scale2), order=1)
        if gt_depth.shape != (img_size, img_size):
            continue
        for j in range(8):
            beta = uniform(0.5, 2)
            tx1 = np.exp(-beta * gt_depth)
            a = 1 - 0.5 * uniform(0, 1)
            m = gt_image.shape[0]
            n = gt_image.shape[1]
            rep_atmosphere = np.tile(np.tile(a, [1, 1, 3]), [m, n, 1])
            tx1 = np.reshape(tx1, [m, n, 1])
            max_transmission = np.tile(tx1, [1, 1, 3])
            haze_image = gt_image * max_transmission + rep_atmosphere * (1 - max_transmission)
            total_num = total_num + 1
            scipy.misc.imsave('a0.9beta1.29.jpg', haze_image)
            scipy.misc.imsave('gt.jpg', gt_image)
            h5f=h5py.File('./'+directory+'/'+str(total_num)+'.h5','w')
            h5f.create_dataset('haze',data=haze_image)
            h5f.create_dataset('trans',data=max_transmission)
            h5f.create_dataset('atom',data=rep_atmosphere)
            h5f.create_dataset('gt',data=gt_image)
split_data('val')

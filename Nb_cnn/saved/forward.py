'''
Description: forward operations for a convolutional neural network

Author: Alejandro Escontrela
Version: 1.0
Date: June 12th, 2018
'''
import numpy as np
import matplotlib.pyplot as plt
from numba import *
from numba import cuda

#####################################################
################ Forward Operations #################
#####################################################
@cuda.jit
def convolution(image, filt, bias, out):
    """
    Given the filter and the image (and a "placeholder" output), returns the image convolluted
    out_dim = int((in_dim - f)/s)+1 ; here s = 1 always
    """
    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x;
    gridY = cuda.gridDim.y * cuda.blockDim.y;
    
    (n_f, n_c_f, f, _) = filt.shape
    in_dim = image.shape[-1]
    for curr_f  in range(n_f):
        b_f = bias[curr_f]
        w_f = filt[curr_f]
        for curr_y in range(startX, in_dim-f+1, gridX):
            for curr_x in range(startY, in_dim-f+1, gridY):
                summ = 0
                im_part = image[:,curr_y:curr_y+f, curr_x:curr_x+f]
                for ii in range(w_f.shape[-2]):
                    for jj in range(w_f.shape[-1]):
                        summ += w_f[0][ii][jj] * im_part[0][ii][jj]
                out[curr_f, curr_y, curr_x] = summ + b_f[0]


@cuda.jit
def maxpool(image, stride ,downsampled):
    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x;
    gridY = cuda.gridDim.y * cuda.blockDim.y;
    
    f = s = stride
    n_c, h_prev, w_prev = image.shape
    for i in range(n_c):
        for curr_y in range(startX, h_prev+1, gridX):
            if curr_y*s + f > h_prev:
                break
            for curr_x in range(startY, w_prev+1, gridY):
                if curr_x*s + f > w_prev:
                    break
                im_part = image[i, curr_y*s:curr_y*s+f, curr_x*s:curr_x*s+f]
                max_valu = im_part[0][0]
                for ii in range(s):
                    for jj in range(s) :
                        if im_part[ii][jj] > max_valu:
                            max_valu = im_part[ii][jj]
                downsampled[i, curr_y, curr_x] = max_valu

def softmax(X):
    out = np.exp(X)
    return out/np.sum(out)

def categoricalCrossEntropy(probs, label):
    return -np.sum(label * np.log(probs))


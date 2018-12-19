'''
Description: forward operations for a convolutional neural network

Author: Alejandro Escontrela
Version: 1.0
Date: June 12th, 2018
'''
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from numba import *
from numba import cuda
from numba import cuda, float32, autojit
import numpy
import math

# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
TPB = 16
"""
Pour le rendu:
    1. Montrer que le réseau marche
    2. Optimiser GRIDDIM et BLOCKDIM empiriquement (trial and error)
        2.1. Selon la taille de l'image en entrée
    3. comparer avec numpy et tensorflow

Citer TOUTES les astuces qu'il a fallu utiliser:
    ne pas trasnposer
    inverser l'ordre des boucles dans le backward conv
    
    

"""
#####################################################
################ Forward Operations #################
#####################################################
@cuda.jit
def convolution(image, filt, bias, out):
    """
    Given the filter (and bias) and the image (and a "placeholder" output), returns the image convolluted with the RELu function applied to it
    out_dim = int((in_dim - f)/s)+1 ; here s = 1 always and f the size of the conv window
    """
    startX, startY, startZ = cuda.grid(3)
    gridX = cuda.gridDim.x * cuda.blockDim.x;
    gridY = cuda.gridDim.y * cuda.blockDim.y;
    gridZ = cuda.gridDim.z * cuda.blockDim.z;
    
    (n_f, n_c_f, f, _) = filt.shape
    in_dim = image.shape[-1]
    for curr_f  in range(startX, n_f, gridX):
        b_f = bias[curr_f]
        w_f = filt[curr_f]
        for curr_y in range(startY, in_dim-f+1, gridY):
            for curr_x in range(startZ, in_dim-f+1, gridZ):
                summ = 0.
                im_part = image[:,curr_y:curr_y+f, curr_x:curr_x+f]
                for kk in range(w_f.shape[-3]):
                    for ii in range(w_f.shape[-2]):
                        for jj in range(w_f.shape[-1]):
                            summ += w_f[kk][ii][jj] * im_part[kk][ii][jj]
                summ += b_f[0]
                ##apply RELu
                if summ >= 0. :
                    out[curr_f, curr_y, curr_x] = summ
                else:
                    out[curr_f, curr_y, curr_x] = 0.


@cuda.jit
def maxpool(image, stride ,downsampled):
    startX, startY, startZ = cuda.grid(3)
    gridX = cuda.gridDim.x * cuda.blockDim.x;
    gridY = cuda.gridDim.y * cuda.blockDim.y;
    gridZ = cuda.gridDim.z * cuda.blockDim.z;
    
    f = s = stride
    n_c, h_prev, w_prev = image.shape
    for i in range(startX, n_c, gridX):
        cuda.syncthreads()
        for curr_y in range(startY, h_prev+1, gridY):
            cuda.syncthreads()
            if not curr_y*s + f > h_prev:
                for curr_x in range(startZ, w_prev+1, gridZ):
                    cuda.syncthreads()
                    if not curr_x*s + f > w_prev:
                        im_part = image[i, curr_y*s:curr_y*s+f, curr_x*s:curr_x*s+f]
                        max_valu = im_part[0][0]
                        cuda.syncthreads()
                        for ii in range(s):
                            for jj in range(s) :
                                if im_part[ii][jj] > max_valu:
                                    max_valu = im_part[ii][jj]
                        cuda.syncthreads()
                        downsampled[i, curr_y, curr_x] = max_valu

def softmax(X):
    out = np.exp(X)
    return out/np.sum(out)

@autojit
def categoricalCrossEntropy(probs, label):
    return -np.sum(label * np.log(probs))


@cuda.jit
def dense_lay(A, B, C, relu):    
    startX = cuda.grid(1)
    gridX = cuda.gridDim.x * cuda.blockDim.x;
    
    for jj in range(startX, C.shape[0] ,gridX):
        summ = 0.
        B_jj = B[jj]
        for kk in range(A.shape[0]):
            summ += A[kk][0] * B_jj[kk]
        ##apply RELu
        if relu:
            if summ <= 0. :
                summ = 0.
        C[jj,0] = summ
            

    
    
    
    
    
    
    
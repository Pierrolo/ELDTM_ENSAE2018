'''
Description: backpropagation operations for a convolutional neural network

Author: Alejandro Escontrela
Version: 1.0
Date: June 12th, 2018
'''
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from numba import *
from numba import cuda
from numba import cuda, float32
import numpy
import math
import numpy as np

from CNN.utils import *

#####################################################
############### Backward Operations #################
#####################################################

@cuda.jit
def back_conv_filt(dconv_prev, conv_in, dfilt):
    startX, startY, startZ = cuda.grid(3);
    gridX = cuda.gridDim.x * cuda.blockDim.x;
    gridY = cuda.gridDim.y * cuda.blockDim.y;
    gridZ = cuda.gridDim.z * cuda.blockDim.z;
    
    (n_f, n_c, f, _) = dfilt.shape
    (_, orig_dim, _) = conv_in.shape
    for curr_f in range(startX, n_f, gridX):
        for curr_c in range(startY, n_c, gridY):
            for ii in range(startZ, f, gridZ):
                for jj in range(f):
                    summ = 0.
                    for out_y in range(orig_dim-f+1):
                            for out_x in range(orig_dim-f+1):
                                summ += dconv_prev[curr_f, out_y, out_x] * conv_in[curr_c,out_y+ii, out_x+jj]
                    dfilt[curr_f,curr_c,ii,jj] = summ
        
@cuda.jit
def back_conv_bias(dconv_prev, dbias):
    startX = cuda.grid(1);
    gridX = cuda.gridDim.x * cuda.blockDim.x;
    
    (n_f, im_dim, _) = dconv_prev.shape
    for curr_f in range(startX, n_f, gridX):
       summ = 0.
       dconv_prev_part = dconv_prev[curr_f]
       for ii in range(im_dim):
           for jj in range(im_dim):
               summ+= dconv_prev_part[ii,jj]
       dbias[curr_f][0] = summ
       
       
@cuda.jit
def backprop_conv(dconv_prev, filt, dout):
    startX, startY, startZ = cuda.grid(3);
    gridX = cuda.gridDim.x * cuda.blockDim.x;
    gridY = cuda.gridDim.y * cuda.blockDim.y;
    gridZ = cuda.gridDim.z * cuda.blockDim.z;
    
    (n_c, n_c_2, f, _) = filt.shape
    (n_f, orig_dim, _) = dout.shape
    for curr_f in range(startX, n_f, gridX):
        for curr_y in range(startY, orig_dim, gridY):
            for curr_x in range(startZ, orig_dim, gridZ):
                summ = 0.
                for kk in range(n_c):
                    for ii in range(f):
                        for jj in range(f):
                            if (curr_y-f+1+ii < dconv_prev.shape[1] and
                                curr_x-f+1+jj < dconv_prev.shape[2] and
                                curr_y-f+1+ii>=0 and
                                curr_x-f+1+jj>=0 and 
                                f-1-ii >=0 and
                                f-1-jj >=0) :
                                summ += dconv_prev[kk, curr_y-f+1+ii, curr_x-f+1+jj] * filt[kk,curr_f,f-1-ii,f-1-jj]
                dout[curr_f, curr_y, curr_x] = summ
                

@cuda.jit
def back_maxpool(dpool, orig, stride, out):
    startX, startY, startZ = cuda.grid(3)
    gridX = cuda.gridDim.x * cuda.blockDim.x;
    gridY = cuda.gridDim.y * cuda.blockDim.y;
    gridZ = cuda.gridDim.z * cuda.blockDim.z;
    
    f = s = stride
    (n_c, orig_dim, _) = orig.shape
    for curr_c in range(startX, n_c, gridX):
        cuda.syncthreads()
        for curr_y in range(startY, orig_dim+1, gridY):
            cuda.syncthreads()
            if not curr_y*s + f > orig_dim:
                for curr_x in range(startZ, orig_dim+1, gridZ):
                    cuda.syncthreads()
                    if not curr_x*s + f > orig_dim:
                        a=b=0
                        max_valu = orig[curr_c, curr_y*s:curr_y*s+f, curr_x*s:curr_x*s+f][0,0]
                        for jj in range(s):
                            for kk in range(s):
                                if max_valu < orig[curr_c, curr_y*s:curr_y*s+f, curr_x*s:curr_x*s+f][jj,kk]:
                                    max_valu = orig[curr_c, curr_y*s:curr_y*s+f, curr_x*s:curr_x*s+f][jj,kk]
                                    a = jj
                                    b = kk
                        out[curr_c, curr_y*s+a, curr_x*s+b] = dpool[curr_c, curr_y, curr_x]


@cuda.jit
def back_dense_lay(A, B, C):
    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x;
    gridY = cuda.gridDim.y * cuda.blockDim.y;
    
    for jj in range(startX, C.shape[0] ,gridX):
        for kk in range(startY, C.shape[1] ,gridY):
            C[jj][kk]= A[jj][0] * B[kk][0]
            
            
@cuda.jit
def back_dense_lay_grad(A, B, C, relu):    
    startX = cuda.grid(1)
    gridX = cuda.gridDim.x * cuda.blockDim.x;
    for jj in range(startX, C.shape[0] ,gridX):
        summ = 0.
        for kk in range(A.shape[0]):
            summ += A[kk][0] * B[kk, jj]
        ##apply RELu
        if relu:
            if summ <= 0. :
                summ = 0.
        C[jj,0] = summ


@cuda.jit
def back_RELu(A, B):
    startX = cuda.grid(1)
    gridX = cuda.gridDim.x * cuda.blockDim.x;
    for ii in range(startX, A.shape[0] ,gridX):
        if A[ii][0] <= 0.:
            B[ii][0] = 0.
            
@cuda.jit
def back_RELu_3d(A, B):
    startX, startY, startZ = cuda.grid(3)
    gridX = cuda.gridDim.x * cuda.blockDim.x;
    gridY = cuda.gridDim.y * cuda.blockDim.y;
    gridZ = cuda.gridDim.z * cuda.blockDim.z;
    for cc in range(startX, A.shape[0] ,gridX):
        for yy in range(startY, A.shape[1] ,gridY):
            for xx in range(startZ, A.shape[2] ,gridZ):
                if A[cc,yy,xx] <= 0.:
                    B[cc,yy,xx] = 0.     
            
            
            
            
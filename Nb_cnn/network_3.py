# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 17:42:13 2018

@author: woill
"""


from Nb_cnn.forward import convolution, maxpool, dense_lay, softmax, categoricalCrossEntropy
from Nb_cnn.backward import back_dense_lay, back_RELu, back_maxpool, back_RELu_3d, back_conv_filt, backprop_conv, back_conv_bias, back_dense_lay_grad
from Nb_cnn.utils import initializeFilter, initializeWeight, extract_data, extract_labels
from Nb_cnn.utils import init_weights, init_outputs_layers, init_weights_gradients, init_outputs_layers_gradient, plot_confusion_matrix
from Nb_cnn.utils import load_weigts, load_outputs_layers, load_weights_gradients, load_outputs_layers_gradient

import numpy as np
from numba import cuda

import math
from numba import *


from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from scipy.misc import ascent
#image = ascent().astype(np.float64)
#image = np.expand_dims(image, axis = 0)
label = np.array([0.,0.,0.,0.,0.,1.,0.,0.,0.,0.])
label = np.expand_dims(label, axis = -1)

m =50000
X = extract_data('train-images-idx3-ubyte.gz', m, 28)
y_dash = extract_labels('train-labels-idx1-ubyte.gz', m).reshape(m,1)
X-= int(np.mean(X))
X/= int(np.std(X))
image = X[0]
image = image.reshape(1, 28, 28)
plt.imshow(image[0])



GRIDDIM= (8, 16, 32)
BLOCKDIM = (2, 4, 32)
NUM_EPOCHS = 5
batch_size = 64
img_dim = 28


#### Def Architecture:

f1 = (4,1,4,4)      ## (output_channel, input_channel, conv_window_h, window_w)
f2 = (8,4,3,3)      ## (output_channel, input_channel, conv_window_h, conv_window_w)
p3 = (2)           ## pooling window = stride
w4 = (64)          ## neurons on dense layer
nb_classes = (10)   ## number of classes to classify on


def forward_n_backward(d_image, label, d_params, d_output_layers, d_weights_gradients, d_outputs_layers_gradient, return_what = "ALL"):
    '''
    Does the forward and backward pass for 1 image; and returns everything
    '''
    res = []
    d_filter_1, d_b1, d_filter_2, d_b2, d_weight_4, d_weight_5 = d_params
    d_out_1, d_out_2, d_out_3, d_out_4, d_out_5, d_flatten = d_output_layers
    d_der_filter_1, d_der_b1, d_der_filter_2, d_der_b2, d_der_weight_4, d_der_weight_5 = d_weights_gradients
    d_der_out_1, d_der_out_2, d_der_out_3, d_der_out_4, d_der_out_5, d_der_flatten, d_der_image = d_outputs_layers_gradient
    
    ### Forward pass:
    convolution[GRIDDIM, BLOCKDIM](d_image, d_filter_1, d_b1, d_out_1)
    convolution[GRIDDIM, BLOCKDIM](d_out_1, d_filter_2, d_b2, d_out_2)
    maxpool[GRIDDIM, BLOCKDIM](d_out_2, p3, d_out_3)
    (nf2, dim2, _) = d_out_3.shape
    d_flatten = d_out_3.reshape((nf2 * dim2 * dim2,1))
    dense_lay[(128), (512)](d_flatten, d_weight_4, d_out_4, True)
    dense_lay[(128), (512)](d_out_4, d_weight_5, d_out_5, False)
    #prediction
    probs = softmax(d_out_5)
    if return_what == "prediction_only":
        res.append(probs)
    
    ### Calcul of the loss # TODO: calculate it in numba 
    loss = categoricalCrossEntropy(probs, label)
    
    ### Backward pass:
    der_out_5 = probs - label # TODO : calculate it in numba
    d_der_out_5 = cuda.to_device(der_out_5)
    
    back_dense_lay[(32,32), (16,32)](d_der_out_5, d_out_4, d_der_weight_5)     ## gradient of the weight of the last layer
    back_dense_lay_grad[(128), (512)](d_der_out_5, d_weight_5, d_der_out_4, False)       ## loss gradient of first dense layer outputs
    back_RELu[(128), (512)](d_out_4, d_der_out_4)                                ## backpropaged through RELu activation function
    
    back_dense_lay[(32,32), (16,32)](d_der_out_4, d_flatten, d_der_weight_4)   ## gradient of the weight of the dense layer after the flatten
    back_dense_lay_grad[(128), (512)](d_der_out_4, d_weight_4, d_der_flatten, False)     ## loss gradient of the dense layer after the flatten
    
    d_der_out_3 = d_der_flatten.reshape(d_out_3.shape) 
    
    back_maxpool[GRIDDIM, BLOCKDIM](d_der_out_3, d_out_2, p3, d_der_out_2)
    back_RELu_3d[GRIDDIM, BLOCKDIM](d_out_2, d_der_out_2)                        ## backpropaged through RELu activation function
    
    back_conv_filt[GRIDDIM, BLOCKDIM](d_der_out_2, d_out_1, d_der_filter_2)
    back_conv_bias[(128),(512)](d_der_out_2, d_der_b2)
    backprop_conv[GRIDDIM, BLOCKDIM](d_der_out_2, d_filter_2, d_der_out_1)
    back_RELu_3d[GRIDDIM, BLOCKDIM](d_out_1, d_der_out_1)   
    
    back_conv_filt[GRIDDIM, BLOCKDIM](d_der_out_1, d_image, d_der_filter_1)
    back_conv_bias[(128),(512)](d_der_out_1, d_der_b1)
    backprop_conv[GRIDDIM, BLOCKDIM](d_der_out_1, d_filter_1, d_der_image)
    
    
    d_params_ret = [d_filter_1, d_b1, d_filter_2, d_b2, d_weight_4, d_weight_5]
    d_output_layers_ret = [d_out_1, d_out_2, d_out_3, d_out_4, d_out_5, d_flatten]
    d_weights_gradients_ret = [d_der_filter_1, d_der_b1, d_der_filter_2, d_der_b2, d_der_weight_4, d_der_weight_5]
    d_outputs_layers_gradient_ret = [d_der_out_1, d_der_out_2, d_der_out_3, d_der_out_4, d_der_out_5, d_der_flatten, d_der_image]
    
    if return_what == "ALL":
        res.append(d_params_ret)
        res.append(d_output_layers_ret)
        res.append(d_weights_gradients_ret)
        res.append(d_outputs_layers_gradient_ret)
        res.append(loss)
        #return [d_params_ret, d_output_layers_ret, d_weights_gradients_ret, d_outputs_layers_gradient_ret, loss]
    if return_what == "weights_grads":
        res.append(d_weights_gradients_ret)
    return res




def train_cnn_nb(lr = 0.035, m = 60000, img_dim = 28, NUM_EPOCHS = 5, batch_size = 32, nb_classes = (10), f1 = (4,1,4,4), f2 = (8,4,3,3), p3 = (2), w4 = (128), test_results = False):
    
    m = m
    if test_results : 
        test_size = 1024
    L_R = lr
    print("Learning Rate =", L_R)
    print("Number of epochs =", NUM_EPOCHS)
    print("Batch Size =", batch_size)
    X = extract_data('train-images-idx3-ubyte.gz', m, img_dim)
    y_dash = extract_labels('train-labels-idx1-ubyte.gz', m).reshape(m,1)
    plt.imshow(X[1].reshape(img_dim,img_dim))
    plt.show()
    #print(y_dash[1])
    X-= int(np.mean(X))
    X/= np.std(X)
    data = np.hstack((X,y_dash))
    np.random.shuffle(data)
    
    if test_results:
        train_data = data[test_size:]
        print("Train Data size =", len(train_data))
        test_data = data[:test_size]
    else :
        train_data = data
        print("Train Data size =", len(train_data))
    
    #### Initialize Everything: 
    pic_dim = img_dim
    params = [filt_1, b1, filt_2, b2, weight_4, weight_5] = init_weights(pic_dim, f1, f2, p3, w4, nb_classes)
    params_shapes = [k.shape for k in params]
    
    ## the rest are just placeholders, since our hand-made functions in numba cannot return anything
    output_layers = [out_1, out_2, out_3, out_4, out_5, flatten] = init_outputs_layers(pic_dim, f1, f2, p3, w4, nb_classes)
    layers_outputs_shapes = [k.shape for k in output_layers]
    
    weights_gradients = [der_filt_1, der_b1, der_filt_2, der_b2, der_weight_4, der_weight_5] = init_weights_gradients(params_shapes)
    der_filt_1_throught_batch, der_b1_throught_batch, der_filt_2_throught_batch, der_b2_throught_batch, der_weight_4_throught_batch, der_weight_5_throught_batch = init_weights_gradients(params_shapes)
    
    image_shape = image.shape
    flat_size = weight_4.shape[1]
    outputs_layers_gradient = [der_image, der_out_1, der_out_2, der_out_3, der_out_4, der_out_5, der_flatten] = init_outputs_layers_gradient(image_shape, layers_outputs_shapes, flat_size)
    
    
    #### Load arrays to device :    
    d_params = [d_filter_1, d_b1, d_filter_2, d_b2, d_weight_4, d_weight_5] = load_weigts(params)
    d_output_layers = [d_out_1, d_out_2, d_out_3, d_out_4, d_out_5, d_flatten] = load_outputs_layers(output_layers)
    d_weights_gradients = [d_der_filter_1, d_der_b1, d_der_filter_2, d_der_b2, d_der_weight_4, d_der_weight_5] = load_weights_gradients(weights_gradients)
    d_outputs_layers_gradient = [d_der_out_1, d_der_out_2, d_der_out_3, d_der_out_4, d_der_out_5, d_der_flatten, d_der_image] = load_outputs_layers_gradient(outputs_layers_gradient)
    
    
    print("Weights shapes =", params_shapes)
    print("Number of weights =", [np.prod(sh) for sh in params_shapes], "Total =", np.sum([np.prod(sh) for sh in params_shapes]))
    #### Forward andbackward pass 
    d_image = cuda.to_device(image)

    
    
    cost = []
    for epoch in range(NUM_EPOCHS):
        np.random.shuffle(train_data)
        batches = [train_data[k:k + batch_size] for k in range(0, train_data.shape[0], batch_size)]
        
        t = tqdm(batches)
        #x, batch = next(enumerate(t))
        for jj, batch in enumerate(t):
            X = batch[:,0:-1] # get batch inputs
            X = X.reshape(len(batch), 1, img_dim, img_dim)
            Y = batch[:,-1] # get batch labels
            [der_filt_1_throught_batch, der_b1_throught_batch, der_filt_2_throught_batch, der_b2_throught_batch, der_weight_4_throught_batch, der_weight_5_throught_batch] = init_weights_gradients(params_shapes)
            cost_ = 0.
            for i in range(len(X)):
                #i = 0
                x = X[i]
                d_image = cuda.to_device(x)
                y = np.eye(nb_classes)[int(Y[i])].reshape(nb_classes, 1) # convert label to one-hot
                
                # Collect Gradients for training
                d_params, d_output_layers, d_weights_gradients, d_outputs_layers_gradient, loss = forward_n_backward(d_image, y, d_params, d_output_layers, d_weights_gradients, d_outputs_layers_gradient, return_what = "ALL")
                
                [d_der_filter_1, d_der_b1, d_der_filter_2, d_der_b2, d_der_weight_4, d_der_weight_5] = d_weights_gradients
                
                der_filt_1_throught_batch += d_der_filter_1
                der_b1_throught_batch += d_der_b1
                der_filt_2_throught_batch += d_der_filter_2
                der_b2_throught_batch += d_der_b2
                der_weight_4_throught_batch += d_der_weight_4
                der_weight_5_throught_batch += d_der_weight_5
                
                cost_+= loss
            filt_1 -= L_R* der_filt_1_throught_batch/len(X)
            b1 -= L_R *der_b1_throught_batch/len(X)
            filt_2 -= L_R *der_filt_2_throught_batch/len(X)
            b2 -= L_R *der_b2_throught_batch/len(X)
            weight_4 -= L_R *der_weight_4_throught_batch/len(X)
            weight_5 -= L_R *der_weight_5_throught_batch/len(X)
            
            params = [filt_1, b1, filt_2, b2, weight_4, weight_5]
            d_params = [d_filter_1, d_b1, d_filter_2, d_b2, d_weight_4, d_weight_5] = load_weigts(params)
            
            cost.append(cost_/len(X))
            t.set_description("Cost: %.8f" % (cost[-1]))
        if test_results:
            ## see the test loss:
            X = test_data[:,0:-1]
            X = X.reshape(len(X), 1, img_dim, img_dim)
            Y = test_data[:,-1] # get batch labels
            cost_test = []
            y_pred = []
            for i in range(len(X)):
                #i = 0
                x = X[i]
                d_image = cuda.to_device(x)
                y = np.eye(nb_classes)[int(Y[i])].reshape(nb_classes, 1) # convert label to one-hot
                pred_prob = forward_n_backward(d_image, y, d_params, d_output_layers, d_weights_gradients, d_outputs_layers_gradient, return_what = "prediction_only")
                loss = categoricalCrossEntropy(pred_prob, y)
                cost_test.append(loss)
                pred = np.argmax(pred_prob)
                y_pred.append(pred)
                if pred-int(Y[i]) != 0 and epoch == NUM_EPOCHS:
                    print("i=",i,"Label :", int(Y[i]), "Pred : ", pred, "Diff :", pred-int(Y[i]))
            
            print("Mean test loss =", np.mean(cost_test))
            plt.figure(figsize = (5,5))
            plt.plot(cost)
            plt.show()
            plt.figure(figsize = (5,5))
            plot_confusion_matrix(confusion_matrix(Y, y_pred), classes=[str(i) for i in range(10)], normalize=True, title='Normalized confusion matrix (Test Set)')
            plt.show()
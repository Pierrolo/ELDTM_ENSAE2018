'''
Description: Utility methods for a Convolutional Neural Network

Author: Alejandro Escontrela
Version: V.1.
Date: June 12th, 2018
'''
from CNN.forward import *
from numba import cuda
import numpy as np
import gzip
import itertools
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix



#####################################################
################## Utility Methods ##################
#####################################################
        
def extract_data(filename, num_images, IMAGE_WIDTH):
    '''
    Extract images by reading the file bytestream. Reshape the read values into a 3D matrix of dimensions [m, h, w], where m 
    is the number of training examples.
    '''
    #print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_WIDTH * IMAGE_WIDTH * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_WIDTH*IMAGE_WIDTH)
        return data

def extract_labels(filename, num_images):
    '''
    Extract label into vector of integer values of dimensions [m, 1], where m is the number of images.
    '''
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

def initializeFilter(size, scale = 1.0):
    stddev = scale/np.sqrt(np.prod(size))
    filt = np.random.normal(loc = 0, scale = stddev, size = size)
    return np.array(filt, dtype= np.float64)

def initializeWeight(size):
    return np.random.standard_normal(size=size) * 0.01

def nanargmax(arr):
    idx = np.nanargmax(arr)
    idxs = np.unravel_index(idx, arr.shape)
    return idxs    



def init_weights(pic_dim, f1, f2, p3, w4, nb_classes):
    filt_1 = initializeFilter(size = f1, scale = 1.0)
    b1 = np.zeros((f1[0],1))#initializeFilter(size = (f1[0],1), scale = 0.01)
    out_dim_1 = pic_dim - f1[-1] + 1
    filt_2 = initializeFilter(size = f2, scale = 1.0)
    b2 = np.zeros((f2[0],1))#initializeFilter(size = (f2[0],1), scale = 0.01)
    out_dim_2 = out_dim_1 - f2[-1] + 1
    f = s = p3
    out_dim_3 = int((out_dim_2 - f)/s)+1
    flat_size = np.prod([f2[0], out_dim_3, out_dim_3])
    weight_4 = initializeWeight((w4, flat_size))
    weight_5 = initializeWeight((nb_classes, w4))
    return [filt_1, b1, filt_2, b2, weight_4, weight_5]


def init_outputs_layers(pic_dim, f1, f2, p3, w4, nb_classes):
    out_dim_1 = pic_dim - f1[-1] + 1
    out_1 = np.zeros((f1[0], out_dim_1, out_dim_1))
    out_dim_2 = out_dim_1 - f2[-1] + 1
    out_2 = np.zeros((f2[0], out_dim_2, out_dim_2))
    f=s=p3
    out_dim_3 = int((out_dim_2 - f)/s)+1    
    out_3 = np.zeros((f2[0], out_dim_3, out_dim_3))
    flatten = np.zeros(np.prod(out_3.shape))
    out_4 = np.zeros((w4, 1))
    out_5 = np.zeros((nb_classes, 1))
    return [out_1, out_2, out_3, out_4, out_5, flatten]


def init_weights_gradients(params_shapes):
    filt_1, b1, filt_2, b2, weight_4, weight_5 = params_shapes
    der_filt_1 = np.zeros(filt_1)
    der_b1 = np.zeros(b1)
    der_filt_2 = np.zeros(filt_2)
    der_b2 = np.zeros(b2)
    der_weight_4 = np.zeros(weight_4)
    der_weight_5 = np.zeros(weight_5)
    return [der_filt_1, der_b1, der_filt_2, der_b2, der_weight_4, der_weight_5]


def init_outputs_layers_gradient(image_shape, layers_outputs_shapes, flat_size):
    out_1, out_2, out_3, out_4, out_5, flatten = layers_outputs_shapes
    der_image = np.zeros(image_shape)
    der_out_1 = np.zeros(out_1)
    der_out_2 = np.zeros(out_2)
    der_out_3 = np.zeros(out_3)
    der_out_4 = np.zeros(out_4)
    der_out_5 = np.zeros(out_5)
    der_flatten = np.zeros((flat_size,1))
    return [der_image, der_out_1, der_out_2, der_out_3, der_out_4, der_out_5, der_flatten]


def load_weigts(params):
    filt_1, b1, filt_2, b2, weight_4, weight_5 = params
    d_filter_1 = cuda.to_device(filt_1)
    d_filter_2 = cuda.to_device(filt_2)
    d_weight_4 = cuda.to_device(weight_4)
    d_weight_5 = cuda.to_device(weight_5)
    d_b1 = cuda.to_device(b1)
    d_b2 = cuda.to_device(b2)
    return [d_filter_1, d_b1, d_filter_2, d_b2, d_weight_4, d_weight_5]


def load_outputs_layers(output_layers):
    out_1, out_2, out_3, out_4, out_5, flatten = output_layers
    d_out_1 = cuda.to_device(out_1)
    d_out_2 = cuda.to_device(out_2)
    d_out_3 = cuda.to_device(out_3)
    d_flatten = cuda.to_device(flatten)
    d_out_4 = cuda.to_device(out_4)
    d_out_5 = cuda.to_device(out_5)
    return [d_out_1, d_out_2, d_out_3, d_out_4, d_out_5, d_flatten]

def load_weights_gradients(weights_gradients):
    der_filt_1, der_b1, der_filt_2, der_b2, der_weight_4, der_weight_5 = weights_gradients
    d_der_weight_5 = cuda.to_device(der_weight_5)
    d_der_weight_4 = cuda.to_device(der_weight_4)
    d_der_filter_2 = cuda.to_device(der_filt_2)
    d_der_filter_1 = cuda.to_device(der_filt_1)
    d_der_b2 = cuda.to_device(der_b2)
    d_der_b1 = cuda.to_device(der_b1)
    return [d_der_filter_1, d_der_b1, d_der_filter_2, d_der_b2, d_der_weight_4, d_der_weight_5]

def load_outputs_layers_gradient(outputs_layers_gradient):
    der_image, der_out_1, der_out_2, der_out_3, der_out_4, der_out_5, der_flatten = outputs_layers_gradient 
    d_der_out_5 = cuda.to_device(der_out_5)
    d_der_out_4 = cuda.to_device(der_out_4)
    d_der_out_3 = cuda.to_device(der_out_3)
    d_der_out_2 = cuda.to_device(der_out_2)
    d_der_out_1 = cuda.to_device(der_out_1)
    d_der_image = cuda.to_device(der_image)
    d_der_flatten = cuda.to_device(der_flatten)
    return [d_der_out_1, d_der_out_2, d_der_out_3, d_der_out_4, d_der_out_5, d_der_flatten, d_der_image]


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        aaaa = 0
        #print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

import os
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.models import Model, save_model
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.utils import plot_model
import numpy as np
from math import exp

def make_lpf(input_shape=(None,None,1)):
    input = Input( input_shape )
    uniform_5_ = Conv2D( 1, kernel_size=(5,5), activation='linear', strides=(1,1), padding='same', name='uniform_5', use_bias=False, trainable=False )
    uniform_5 = uniform_5_( input )
    uniform_7_ = Conv2D( 1, kernel_size=(7,7), activation='linear', strides=(1,1), padding='same', name='uniform_7', use_bias=False, trainable=False )
    uniform_7 = uniform_7_( input )
    gaussian_20_ = Conv2D( 1, kernel_size=(33,33), activation='linear', strides=(1,1), padding='same', name='gaussian_20', use_bias=False, trainable=False )
    gaussian_20 = gaussian_20_( input )
    gaussian_30_ = Conv2D( 1, kernel_size=(33,33), activation='linear', strides=(1,1), padding='same', name='gaussian_30', use_bias=False, trainable=False )
    gaussian_30 = gaussian_30_( input )
    output = concatenate( [uniform_5, uniform_7, gaussian_20, gaussian_30] )
    return Model( input, output )

def kernel_uniform( size ):
    kernel = np.ones( (size, size) ) / ( size*size )
    kernel = kernel.reshape( (size, size, 1, 1) )
    return kernel

def kernel_gaussian( sigma ):
    kernel = np.zeros( (33, 33) )
    for r in range( 33 ):
        for c in range( 33 ):
            kernel[r][c] = exp( -((r-16)^2+(c-16)^2)/(sigma+sigma) )
    kernel /= np.sum( kernel )
    kernel = kernel.reshape( (33, 33, 1, 1) )
    return kernel

def merge_model( denoising_model_path ):
    #make lpf
    lpf_model = make_lpf()
    lpf_model.get_layer('uniform_5').set_weights([kernel_uniform(5),])
    lpf_model.get_layer('uniform_7').set_weights([kernel_uniform(7),])
    lpf_model.get_layer('gaussian_20').set_weights([kernel_gaussian(20),])
    lpf_model.get_layer('gaussian_30').set_weights([kernel_gaussian(30),])
    lpf_model.save( './lpf.model' )

    denoising_model = load_model( denoising_model_path )

    input = Input( (None, None, 1) )
    lpf = lpf_model( input )
    denoising = denoising_model( lpf )

    merged_model = Model( input, denoising )
    merged_model.save( '/data/model/saved_model/denoiser_merged_13_any_size.model' )
    merged_model.save_weights( '/data/model/saved_model/denoiser_merged_13_any_size.weight' )

if __name__ == '__main__':
    lpf_model = make_lpf(input_shape=(512, 512, 1))
    lpf_model.summary()
    plot_model(lpf_model, 'lpf_model.png', show_shapes=True, rankdir='TB')


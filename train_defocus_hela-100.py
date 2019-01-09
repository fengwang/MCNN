from model import build_model

import copy
import os
import glob
import imageio
import numpy as np
import random
from scipy.signal import convolve
from skimage.measure import block_reduce
from keras.utils import multi_gpu_model
from keras.models import load_model
from imageio import imsave
from keras.optimizers import Adam
from keras import backend as K

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# /data/cache/astigmatism_16_1.npz
def report( array, comment ):
    print( f'{comment}: size - {array.shape}, min - {np.amin(array)}, max - {np.amax(array)}, mean - {np.mean(array)}, variance - {np.var(array)}, std - {np.std(array)}.' )

def make_block_reduce( input_layers, dim=(2,2), mode=np.mean ):
    stacked_layers = [ block_reduce( image, dim, mode ) for image in input_layers ]
    return np.asarray( stacked_layers, dtype='float32' )

# defocus at -10 \mu m
def prepare_defocus_data( path='/data/cache/defocus_1_mandelbrot-gaussian-10db-10.282856.npz'):
    shape=(512, 512)
    data = np.load(path)
    input_layers = copy.deepcopy( data['sim'] )
    outputs = copy.deepcopy( data['orig'] )
    n, row, col, *_ = outputs.shape
    input_layers = input_layers.reshape( (n, row, col, 1) )
    input_layers = ( input_layers-np.mean(input_layers) ) / (np.std(input_layers) + 1.0e-10)
    #assert ch==1, 'channel of outputs should be 1 for astig'
    output_images = outputs.reshape( (n, row, col) )
    output_images = (output_images-np.amin(output_images)) / ( np.amax(output_images)-np.amin(output_images)+1.0e-10)
    output_layers = [np.asarray( output_images, dtype='float32' ), None, None, None, None, None, None, None]
    for idx in range( 7 ):
        output_layers[idx+1] = make_block_reduce( output_layers[idx], (2,2), np.mean )
        output_layers[idx] = output_layers[idx].reshape( output_layers[idx].shape + (1,) )
    output_layers[7] = output_layers[7].reshape( output_layers[7].shape + (1,) )
    imsave( 'input_0.png', input_layers[0].reshape( shape ) )
    imsave( 'output_0.png', output_layers[0][0].reshape( shape ) )
    imsave( 'output_1.png', output_layers[1][0].reshape( (256, 256) ) )

    data = None

    return [input_layers, output_layers]


def train_mdcnn( model_path='./model/MDCNN-III-hela-I-100.h5', image_shape=(512, 512), epochs=1024, batch_size=8, gpus=0, index=0, input_layers=None, output_layers=None ):

    if os.path.isfile( model_path ):
        mdcnn = load_model( model_path )
        print( f'loading MDCNN model from {model_path}' )
    else:
        mdcnn = build_model()

    n_images, *_ = input_layers.shape
    print( f'training dataset generated, with {n_images} input images all of shape {image_shape}' )

    if gpus > 1:
        mdcnn = multi_gpu_model( mdcnn, gpus=gpus )
    print( f'MDCNN-I training with {n_images} images of {epochs} epochs with a batch size {batch_size} and {gpus} GPUs.' )

    i_image = imageio.imread( '/data/experimental/Feng2018/HeLa_cell_focal_series/Pos0/img_000000000_Default0_015.tif' )
    r, c = i_image.shape
    offset_r, offset_c = (r-2048)>>1, (c-2048)>>1
    i_image = i_image[ offset_r:offset_r+2048, offset_c:offset_c+2048]
    i_image = (i_image-np.mean(i_image)) / ( np.std(i_image)+1.0e-10 )
    row, col = i_image.shape
    input_image = i_image.reshape( (1, row, col, 1) )

    i_image_se = imageio.imread( '/data/experimental/Feng2018/HeLa_cell_focal_series/Pos0/img_000000000_Default0_035.tif' )
    r, c = i_image_se.shape
    offset_r, offset_c = (r-2048)>>1, (c-2048)>>1
    i_image_se = i_image_se[ offset_r:offset_r+2048, offset_c:offset_c+2048]
    i_image_se = (i_image_se-np.mean(i_image_se)) / ( np.std(i_image_se)+1.0e-10 )
    row, col = i_image_se.shape
    input_image_se = i_image_se.reshape( (1, row, col, 1) )

    lr = 0.001
    for idx in range( epochs ):
        opt = Adam(lr=lr)
        lr = lr * 0.9999
        mdcnn.compile( loss='mae', optimizer=opt )
        mdcnn.fit( input_layers, output_layers, batch_size=batch_size, epochs=1, verbose=2, validation_split=0.125 )
        mdcnn.save( f'./hela_defocus-1/current_model_{index}_{idx}.model' )

        output_image, *_ = mdcnn.predict( input_image )
        output_image = output_image.reshape( (row, col) )
        imageio.imsave( f'./hela_defocus-1/{index}_{idx}_prediction.png', output_image )

        output_image_se, *_ = mdcnn.predict( input_image_se )
        output_image_se = output_image_se.reshape( (row, col) )
        imageio.imsave( f'./hela_defocus-1/{index}_{idx}_prediction_se.png', output_image_se )

    mdcnn.save( model_path )
    K.clear_session()

if __name__ == '__main__':
    #train_mdcnn( gpus=1, epochs=100, batch_size=4, model_path = './hela_defocus-1/current_model.model' )
    input_layers, output_layers = prepare_defocus_data()
    print( 'training data loaded' )
    for idx in range( 100 ):
        train_mdcnn( gpus=1, epochs=2, batch_size=4, model_path = './hela_defocus-1/current_model_-10.model', index=idx, input_layers=input_layers, output_layers=output_layers )


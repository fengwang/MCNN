from model import build_model

import os
import glob
import imageio
import numpy as np
import random
from scipy.signal import convolve
from skimage.measure import block_reduce
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.models import load_model
from imageio import imsave
from keras.callbacks import TensorBoard

# /data/cache/astigmatism_16_1.npz
def report( array, comment ):
    print( f'{comment}: size - {array.shape}, min - {np.amin(array)}, max - {np.amax(array)}, mean - {np.mean(array)}, variance - {np.var(array)}, std - {np.std(array)}.' )

def make_block_reduce( input_layers, dim=(2,2), mode=np.mean ):
    stacked_layers = [ block_reduce( image, dim, mode ) for image in input_layers ]
    return np.asarray( stacked_layers, dtype='float32' )

def prepare_astig_data( path='/data/cache/astigmatism_16_1.npz'):
    shape=(512, 512)
    data = np.load(path)
    input_layers = data['input_16']
    input_layers = ( input_layers-np.mean(input_layers) ) / (np.std(input_layers) + 1.0e-10)
    outputs = data['output_1']
    n, row, col, ch = outputs.shape
    assert ch==1, 'channel of outputs should be 1 for astig'
    output_images = outputs.reshape( (n, row, col) )
    output_images = (output_images-np.amin(output_images)) / ( np.amax(output_images)-np.amin(output_images)+1.0e-10)
    output_layers = [np.asarray( output_images, dtype='float32' ), None, None, None, None, None, None, None]
    for idx in range( 7 ):
        output_layers[idx+1] = make_block_reduce( output_layers[idx], (2,2), np.mean )
        output_layers[idx] = output_layers[idx].reshape( output_layers[idx].shape + (1,) )
    output_layers[7] = output_layers[7].reshape( output_layers[7].shape + (1,) )
    imsave( 'input_0.png', input_layers[0, :, :, 0].reshape( shape ) )
    imsave( 'output_0.png', output_layers[0][0].reshape( shape ) )
    imsave( 'output_1.png', output_layers[1][0].reshape( (256, 256) ) )
    report( input_layers, 'input_layers' )
    report( output_layers[0], 'out_0' )
    report( output_layers[1], 'out_1' )
    report( output_layers[2], 'out_2' )
    report( output_layers[3], 'out_3' )
    report( output_layers[4], 'out_4' )
    report( output_layers[5], 'out_5' )
    report( output_layers[6], 'out_6' )
    report( output_layers[7], 'out_7' )

    return [input_layers, output_layers]


def train_mdcnn( model_path='./model/MDCNN-IV.h5', image_shape=(512, 512), epochs=1024, batch_size=8, gpus=0 ):

    if os.path.isfile( model_path ):
        mdcnn = load_model( model_path )
        print( f'loading MDCNN model from {model_path}' )
    else:
        mdcnn = build_model(img_channels=16)

    input_layers, output_layers = prepare_astig_data( path='/data/cache/astigmatism_16_1.npz')
    n_images, *_ = input_layers.shape
    print( f'training dataset generated, with {n_images} input images all of shape {image_shape}' )

    tensor_board = TensorBoard(log_dir='./log', histogram_freq=0, write_graph=True, write_images=True)

    if gpus > 1:
        mdcnn = multi_gpu_model( mdcnn, gpus=gpus )
    print( f'MDCNN-IV training with {n_images} images of {epochs} epochs with a batch size {batch_size} and {gpus} GPUs.' )
    mdcnn.compile( loss='mae', optimizer='adam' )
    #optimizer = Adam( decay=0.01 )
    #mdcnn.compile( loss='mae', optimizer=optimizer )
    mdcnn.fit( input_layers, output_layers, batch_size=batch_size, epochs=epochs, verbose=1,validation_split=0.125, callbacks=[tensor_board] )
    mdcnn.save( model_path )

    groundtruth_output = output_layers[0]
    mdcnn_output, *_ = mdcnn.predict( input_layers )
    for idx in range( n_images ):
        ground_truth =  groundtruth_output[idx].reshape( image_shape )
        prediction = mdcnn_output[idx].reshape( image_shape )
        df = np.sum( np.abs( ground_truth - prediction )) / ( 512.0 * 512.0 )
        print( f'saving validation images for index {idx}', end='\r' )
        imsave( f'./validation_images_astig_16/{idx}_input_0.jpg', input_layers[idx, :, :, 0].reshape( image_shape ) )
        imsave( f'./validation_images_astig_16/{idx}_input_1.jpg', input_layers[idx, :, :, 1].reshape( image_shape ) )
        imsave( f'./validation_images_astig_16/{idx}_input_2.jpg', input_layers[idx, :, :, 2].reshape( image_shape ) )
        imsave( f'./validation_images_astig_16/{idx}_ground.jpg', groundtruth_output[idx].reshape( image_shape ) )
        imsave( f'./validation_images_astig_16/{idx}_mdcnn_{df}.jpg', mdcnn_output[idx].reshape( image_shape ) )

if __name__ == '__main__':
    train_mdcnn( gpus=2, epochs=2048, batch_size=32 )


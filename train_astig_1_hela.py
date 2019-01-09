from model import build_model

import copy
import os
import glob
import imageio
import numpy as np
import random
from scipy.signal import convolve
from skimage.measure import block_reduce
from keras.callbacks import TensorBoard
from keras.utils import multi_gpu_model
from keras.models import load_model
from imageio import imsave
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import keras.backend as K

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# /data/cache/astigmatism_16_1.npz
def report( array, comment ):
    print( f'{comment}: size - {array.shape}, min - {np.amin(array)}, max - {np.amax(array)}, mean - {np.mean(array)}, variance - {np.var(array)}, std - {np.std(array)}.' )

def make_block_reduce( input_layers, dim=(2,2), mode=np.mean ):
    stacked_layers = [ block_reduce( image, dim, mode ) for image in input_layers ]
    return np.asarray( stacked_layers, dtype='float32' )

def trim_image_from_path( image_path, shape ):
    image = imageio.imread( image_path )
    image = np.asarray( image, dtype='float32' )
    #image = (image-np.amin(image))/(np.amax(image)-np.amin(image)+1.0e-10)
    image = (image+0.5) / 256.0

    if 3 == len(image.shape): # rgb -> gray
        image = 0.2627*image[:,:,0]+0.6780*image[:,:,1]+0.0593*image[:,:,2]

    scaling_ratio = [ int(a/b) for a, b in zip( image.shape, shape ) ]
    if scaling_ratio[0] > 1 and scaling_ratio[1] > 1 :
        image = block_reduce( image, tuple(scaling_ratio), np.mean )

    dim_diff = [ (a - b)>>1 for a, b in zip( image.shape, shape ) ]
    if dim_diff[0] < 0 or dim_diff[1] < 0:
        print( f'size too small? dropping image {image_path}' )
        return None
    print( f'trimmed image {image_path}' )
    return image[dim_diff[0]:dim_diff[0]+shape[0], dim_diff[1]:dim_diff[1]+shape[1]]

def prepare_astig_data( path='/data/cache/astigmatism_hela_33_1.npz'):
    shape=(512, 512)
    data = np.load(path)
    #input_layers = data['input_1']
    input_layers = copy.deepcopy( data['input_1'] )
    input_layers = ( input_layers-np.mean(input_layers) ) / (np.std(input_layers) + 1.0e-10)
    #input_layers = ( input_layers-np.amin(input_layers) ) / (np.amax(input_layers) - np.amin(input_layers) + 1.0e-10)
    #outputs = data['output_1']
    outputs = copy.deepcopy( data['output_1'] )
    n, row, col, ch = outputs.shape
    assert ch==1, 'channel of outputs should be 1 for astig'
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

input_layers, output_layers =  (None, None)
i_image = None

def train_mdcnn( model_path='./model/MDCNN-III-hela-I.h5', image_shape=(512, 512), epochs=1024, batch_size=8, gpus=0, train_index=0 ):

    if os.path.isfile( model_path ):
        mdcnn = load_model( model_path )
        print( f'loading MDCNN model from {model_path}' )
    else:
        mdcnn = build_model()

    ##
    #input_layers, output_layers = prepare_astig_data( path='/data/cache/astigmatism_hela_33_1.npz')
    #input_layers, output_layers = prepare_astig_data( path='/data/cache/astigmatism_hela_33_2.npz')
    #input_layers, output_layers = prepare_astig_data( path='/data/cache/astigmatism_hela_33_3.npz')
    global input_layers
    global output_layers
    if input_layers is None or output_layers is None:
        input_layers, output_layers = prepare_astig_data( path='/data/cache/astigmatism_hela_33_4.npz')

    n_images, *_ = input_layers.shape
    print( f'training dataset generated, with {n_images} input images all of shape {image_shape}' )

    tensor_board = TensorBoard(log_dir='./log', histogram_freq=0, write_graph=True, write_images=True)

    if gpus > 1:
        mdcnn = multi_gpu_model( mdcnn, gpus=gpus )
    print( f'MDCNN-I training with {n_images} images of {epochs} epochs with a batch size {batch_size} and {gpus} GPUs.' )
    '''
    mdcnn.compile( loss='mae', optimizer=Adam(decay=1e-6) )
    mdcnn.fit( input_layers, output_layers, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.25, callbacks=[tensor_board] )
    mdcnn.save( model_path )
    '''

    global i_image
    if i_image is None:
        i_image = imageio.imread( '/data/experimental/Feng2018/HeLa_cell_astig_series/t_0.png' )
    i_image = (i_image-np.mean(i_image)) / ( np.std(i_image)+1.0e-10 )
    #i_image = (i_image-np.amin(i_image)) / ( np.amax(i_image)-np.amin(i_image)+1.0e-10 )
    row, col = i_image.shape
    input_image = i_image.reshape( (1, row, col, 1) )

    lr = 0.001
    for idx in range( epochs ):
        opt = Adam(lr=lr)
        lr = lr * 0.9999
        mdcnn.compile( loss='mae', optimizer=opt )
        #mdcnn.compile( loss='mse', optimizer=opt )
        mdcnn.fit( input_layers, output_layers, batch_size=batch_size, epochs=1, verbose=2, validation_split=0.125 )
        #mdcnn.save( f'./hela_astig_1/current_model_{idx}.model' )
        output_image, *_ = mdcnn.predict( input_image )
        output_image = output_image.reshape( (row, col) )
        imageio.imsave( f'./hela_astig_1/{idx}_{train_index}_prediction.png', output_image )

    mdcnn.save( model_path )
    K.clear_session()

    '''
    groundtruth_output = output_layers[0]
    mdcnn_output, *_ = mdcnn.predict( input_layers )
    for idx in range( n_images ):
        ground_truth =  groundtruth_output[idx].reshape( image_shape )
        prediction = mdcnn_output[idx].reshape( image_shape )
        df = np.sum( np.abs( ground_truth - prediction )) / ( 512.0 * 512.0 )
        print( f'saving validation images for index {idx}', end='\r' )
        imsave( f'./validation_images_astig_1/{idx}_input.jpg', input_layers[idx].reshape( image_shape ) )
        imsave( f'./validation_images_astig_1/{idx}_ground.jpg', groundtruth_output[idx].reshape( image_shape ) )
        imsave( f'./validation_images_astig_1/{idx}_mdcnn_{df}.jpg', mdcnn_output[idx].reshape( image_shape ) )
    '''

if __name__ == '__main__':
    for train_index in range( 1000 ):
        train_mdcnn( gpus=1, epochs=1, batch_size=2, model_path = './hela_astig_1/current_model.model', train_index=train_index )


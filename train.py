from model import build_model

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

def report( array, comment ):
    print( f'{comment}: size - {array.shape}, min - {np.amin(array)}, max - {np.amax(array)}, mean - {np.mean(array)}, variance - {np.var(array)}, std - {np.std(array)}.' )

def make_block_reduce( input_layers, dim=(2,2), mode=np.mean ):
    stacked_layers = [ block_reduce( image, dim, mode ) for image in input_layers ]
    return np.asarray( stacked_layers, dtype='float32' )

def trim_image_from_path( image_path, shape ):
    image = imageio.imread( image_path )
    image = np.asarray( image, dtype='float32' )
    #image = (image-np.amin(image))/(np.amax(image)-np.amin(image)+1.0e-10)
    image = image / 255.0

    if 3 == len(image.shape): # rgb -> gray
        image = 0.2627*image[:,:,0]+0.6780*image[:,:,1]+0.0593*image[:,:,2]

    scaling_ratio = [ int(a/b) for a, b in zip( image.shape, shape ) ]
    if scaling_ratio[0] > 1 and scaling_ratio[1] > 1 :
        image = block_reduce( image, tuple(scaling_ratio), np.mean )

    dim_diff = [ (a - b)>>1 for a, b in zip( image.shape, shape ) ]
    if dim_diff[0] <= 0 or dim_diff[1] <= 0:
        return None
    return image[dim_diff[0]:dim_diff[0]+shape[0], dim_diff[1]:dim_diff[1]+shape[1]]

def prepare_data( paths, shape, number ):
    random.shuffle( paths )
    output_images = []
    input_images = []
    kernel = np.asarray( [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype='float32' )

    counter = 0
    for path in paths:
        trimmed_image = trim_image_from_path(path, shape )
        if trimmed_image is not None:
            print( f'preparing training set from {path} - {counter}/{number}', end='\r' )
            output_images.append(trimmed_image)
            input_images.append( convolve( trimmed_image, kernel, mode='same' ) )
            counter += 1
        if counter == number:
            break

    input_layers = np.asarray( input_images, dtype='float32' ).reshape( (number,)+shape+(1,) )
    input_layers = ( input_layers-np.mean(input_layers) ) / (np.std(input_layers) + 1.0e-10)

    output_layers = [np.asarray( output_images, dtype='float32' ), None, None, None, None, None, None, None]
    for idx in range( 7 ):
        output_layers[idx+1] = make_block_reduce( output_layers[idx], (2,2), np.mean )
        output_layers[idx] = output_layers[idx].reshape( output_layers[idx].shape + (1,) )
    output_layers[7] = output_layers[7].reshape( output_layers[7].shape + (1,) )

    imsave( 'input_0.png', input_layers[0].reshape( shape ) )
    imsave( 'output_0.png', output_layers[0][0].reshape( shape ) )
    imsave( 'output_1.png', output_layers[1][0].reshape( (256, 256) ) )

    report( input_layers, 'input_layers' )
    report( output_layers[0], 'output_layers[0]' )
    report( output_layers[1], 'output_layers[1]' )
    report( output_layers[2], 'output_layers[2]' )
    report( output_layers[3], 'output_layers[3]' )
    report( output_layers[4], 'output_layers[4]' )
    report( output_layers[5], 'output_layers[5]' )
    report( output_layers[6], 'output_layers[6]' )

    return [input_layers, output_layers]

def train_mdcnn( model_path='./model/MDCNN-I.h5', image_shape=(512, 512), n_images=1024, epochs=1024, batch_size=8, gpus=0 ):

    if os.path.isfile( model_path ):
        mdcnn = load_model( model_path )
        print( f'loading MDCNN model from {model_path}' )
    else:
        mdcnn = build_model()

    input_layers, output_layers = prepare_data( glob.glob( './images/*/*.jpg'), image_shape, n_images )
    print( f'training dataset generated, with {n_images} input images all of shape {image_shape}' )

    tensor_board = TensorBoard(log_dir='./log', histogram_freq=0, write_graph=True, write_images=True)

    if gpus > 1:
        mdcnn = multi_gpu_model( mdcnn, gpus=gpus )
    print( f'MDCNN-I training with {n_images} images of {epochs} epochs with a batch size {batch_size} and {gpus} GPUs.' )
    mdcnn.compile( loss='mae', optimizer='adam' )
    mdcnn.fit( input_layers, output_layers, batch_size=batch_size, epochs=epochs, verbose=1,validation_split=0.25, callbacks=[tensor_board] )
    mdcnn.save( model_path )

    groundtruth_output = output_layers[0]
    mdcnn_output, *_ = mdcnn.predict( input_layers )
    for idx in range( n_images - (n_images>> 2), n_images ):
        print( f'saving validation images for index {idx}', end='\r' )
        imsave( f'./validation_images/{idx}_input.jpg', input_layers[idx].reshape( image_shape ) )
        imsave( f'./validation_images/{idx}_ground_truth.jpg', groundtruth_output[idx].reshape( image_shape ) )
        imsave( f'./validation_images/{idx}_mdcnn.jpg', mdcnn_output[idx].reshape( image_shape ) )

if __name__ == '__main__':
    train_mdcnn( gpus=2, n_images=1024, epochs=128, batch_size=8 )


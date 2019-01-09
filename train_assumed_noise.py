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

import sys
sys.path.insert(0, "/home/feng/python/modules/feng/")
from generators import fake_images

def sim_noise_poission( image ):
    image = ( image - np.amin(image) )/ ( np.amax(image) - np.amin(image) + 1.0e-10 )
    image = (image + np.random.random(image.shape))/2.0
    level = random.random() * 9.00 + 1.0;
    image = np.random.poisson(image*level ) / level
    satuaration_level = random.random()*0.1+0.9
    row, col = image.shape
    for r in range( row ):
        for c in range( col ):
            if image[r][c] > satuaration_level:
                image[r][c] = satuaration_level
            if image[r][c] < 0.0:
                image[r][c] = 0.0
    image = ( image - np.amin(image) )/ ( np.amax(image) - np.amin(image) + 1.0e-10 )
    return image

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
    if dim_diff[0] < 0 or dim_diff[1] < 0:
        print( f'size too small? dropping image {image_path}' )
        return None
    print( f'trimmed image {image_path}' )
    return image[dim_diff[0]:dim_diff[0]+shape[0], dim_diff[1]:dim_diff[1]+shape[1]]

def prepare_data( paths, shape, number ):
    random.shuffle( paths )
    output_images = []
    input_images = []
    kernel = np.asarray( [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype='float32' )

    counter = 0
    for path in paths:
        print( f'preparing data from {path}', end='\r' )
        trimmed_image = trim_image_from_path(path, shape )
        if trimmed_image is not None:
            print( f'preparing training set from {path} - {counter}/{number}', end='\r' )
            output_images.append(trimmed_image)
            #input_images.append( convolve( trimmed_image, kernel, mode='same' ) )
            input_images.append( sim_noise_poission( trimmed_image ) )
            counter += 1
            print( f'got an image from {path}', end='\r' )
        if counter == number:
            break
    fakes = fake_images.fake_images(number, shape, sigmas=(1,10), range_of_columns=(50, 200), pixel_interval=12, max_intensity=5)
    for idx in range(number):
        output_images.append( fakes[idx] )
        input_images.append( sim_noise_poission(fakes[idx]) )

    input_layers = np.asarray( input_images, dtype='float32' ).reshape( (number+number,)+shape+(1,) )
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

    test_image = imageio.imread( './data/direct_groundtruth_512.png' )
    noised_img = sim_noise_poission( test_image )
    imageio.imsave( './data/noisy.png', noised_img )
    noised_img = noised_img.reshape( ( 1, 512, 512, 1 ) )

    if os.path.isfile( model_path ):
        mdcnn = load_model( model_path )
        print( f'loading MDCNN model from {model_path}' )
    else:
        mdcnn = build_model()

    #input_layers, output_layers = prepare_data( glob.glob( './images/*/*.jpg'), image_shape, n_images )
    input_layers, output_layers = prepare_data( glob.glob( '/data/mandelbrot/*/*.png'), image_shape, n_images )
    print( f'training dataset generated, with {n_images} input images all of shape {image_shape}' )

    tensor_board = TensorBoard(log_dir='./log', histogram_freq=0, write_graph=True, write_images=True)

    if gpus > 1:
        mdcnn = multi_gpu_model( mdcnn, gpus=gpus )
    print( f'MDCNN-I training with {n_images} images of {epochs} epochs with a batch size {batch_size} and {gpus} GPUs.' )
    mdcnn.compile( loss='mae', optimizer='adam' )
    for ep in range( epochs ):
        mdcnn.fit( input_layers, output_layers, batch_size=batch_size, epochs=2, verbose=1,validation_split=0.25, callbacks=[tensor_board] )
        mdcnn_output, *_ = mdcnn.predict( noised_img )
        o_image = mdcnn_output.reshape( (512, 512) )
        imageio.imsave( f'./data/reconstructed_{ep}.png', o_image )

    mdcnn.save( model_path )

if __name__ == '__main__':
    train_mdcnn( gpus=2, n_images=512, epochs=1024, batch_size=8 )


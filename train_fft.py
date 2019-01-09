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

def normalize( array ):
    return ( array - np.amin(array) ) / ( np.amax(array) - np.amin(array) + 1.0e-10 )

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

def make_fft( input_images ):
    real, imag = normalize(input_images[:,:,0]) * 2.0 - 1.0, normalize( input_images[:,:,1] ) * 2.0 - 1.0
    phase_image = real + 1.0j * imag
    row, col = phase_image.shape
    transform = np.fft.fft2( phase_image ) / ( row * col * 1.0 )
    twin_images = np.zeros((row,col,2))
    #twin_images[:,:,0] = np.real( transform )
    #twin_images[:,:,1] = np.imag( transform )
    twin_images[:,:,0] = normalize( np.angle( transform ) )
    twin_images[:,:,1] = normalize( np.abs( transform ) )
    return twin_images

def prepare_data( paths, shape, number ):
    random.shuffle( paths )
    sampled_images = []

    counter = 0
    for path in paths:
        print( f'preparing data from {path}', end='\r' )
        trimmed_image = trim_image_from_path(path, shape )
        if trimmed_image is not None:
            print( f'preparing training set from {path} - {counter}/{number}', end='\r' )
            trimmed_image = normalize( trimmed_image )
            sampled_images.append(trimmed_image)
            counter += 1
            print( f'got an image from {path}', end='\r' )
        if counter == (number+number):
            break

    print( f'{number*2} images retrieved with shape {shape}' )

    input_images = []
    output_images = []
    for idx in range( number ):
        twin_images = np.zeros( (shape+(2,)) )
        twin_images[:,:,0] = sampled_images[idx+idx]
        twin_images[:,:,1] = sampled_images[idx+idx+1]
        input_images.append( twin_images )
        output_images.append( make_fft(twin_images) )

    print( f'{number} images proceeded' )

    #input_layers = np.asarray( input_images, dtype='float32' )
    input_layers = np.asarray( output_images, dtype='float32' )
    #
    #output_layers = [np.asarray( output_images, dtype='float32' ), None, None, None, None, None, None, None]
    output_layers = [np.asarray( input_images, dtype='float32' ), None, None, None, None, None, None, None]

    for idx in range( 7 ):
        output_layers[idx+1] = make_block_reduce( output_layers[idx], (2,2,1), np.mean )

    report( input_layers, 'input_layers' )
    report( output_layers[0], 'output_layers[0]' )
    report( output_layers[1], 'output_layers[1]' )
    report( output_layers[2], 'output_layers[2]' )
    report( output_layers[3], 'output_layers[3]' )
    report( output_layers[4], 'output_layers[4]' )
    report( output_layers[5], 'output_layers[5]' )
    report( output_layers[6], 'output_layers[6]' )

    imsave( 'input_0.png', input_layers[0,:,:,0] )
    imsave( 'output_0.png', output_layers[0][0,:,:,0] )
    imsave( 'output_1.png', output_layers[1][0,:,:,0] )

    return [input_layers, output_layers]

def train_mdcnn( model_path='./model/MDCNN-fft.h5', image_shape=(512, 512), n_images=1024, epochs=1024, batch_size=8, gpus=0 ):

    if os.path.isfile( model_path ):
        mdcnn = load_model( model_path )
        print( f'loading MDCNN model from {model_path}' )
    else:
        mdcnn = build_model( img_channels=2, output_channels=2, output_activation='sigmoid' )

    #input_layers, output_layers = prepare_data( glob.glob( '/data/mandelbrot/*/*.png'), image_shape, n_images )
    input_layers, output_layers = prepare_data( glob.glob( '/data/wallpapers/*/*.jpg'), image_shape, n_images )
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

    for idx in range( n_images ):
        ground_truth =  groundtruth_output[idx,:,:,0].reshape( image_shape )
        prediction = mdcnn_output[idx,:,:,0].reshape( image_shape )
        df = np.sum( np.abs( ground_truth - prediction )) / ( 512.0 * 512.0 )
        print( f'saving validation images for index {idx}', end='\r' )
        imsave( f'./validation_images/{idx}_input.jpg', normalize(input_layers[idx,:,:,0].reshape( image_shape ))*255.0 )
        imsave( f'./validation_images/{idx}_ground.jpg', normalize(groundtruth_output[idx,:,:,0].reshape( image_shape ))*255.0 )
        imsave( f'./validation_images/{idx}_mdcnn_{df}.jpg', normalize(mdcnn_output[idx,:,:,0].reshape( image_shape ))*255.0 )
        imsave( f'./validation_images/{idx}_input_1.jpg', normalize(input_layers[idx,:,:,1].reshape( image_shape ))*255.0 )
        imsave( f'./validation_images/{idx}_ground_1.jpg', normalize(groundtruth_output[idx,:,:,1].reshape( image_shape ))*255.0 )
        imsave( f'./validation_images/{idx}_mdcnn_{df}_1.jpg', normalize(mdcnn_output[idx,:,:,1].reshape( image_shape ))*255.0 )

if __name__ == '__main__':
    train_mdcnn( gpus=2, n_images=1024, epochs=800, batch_size=8 )


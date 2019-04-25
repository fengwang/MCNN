testing_data_path = '/data2/feng/door_mirror/dataset/set4/0_256_screens_cameras.npz_1280X720_scaled_to_0_1.npz_flipped_cropped_0_1.npz'
training_set_path = '/data2/feng/door_mirror/dataset/set4/1_256_screens_cameras.npz_1280X720_scaled_to_0_1.npz_flipped_cropped_0_1.npz'
training_set_path_se = '/data2/feng/door_mirror/dataset/set4/2_256_screens_cameras.npz_1280X720_scaled_to_0_1.npz_flipped_cropped_0_1.npz'
training_set_path_rd = '/data2/feng/door_mirror/dataset/set4/3_256_screens_cameras.npz_1280X720_scaled_to_0_1.npz_flipped_cropped_0_1.npz'
storage_path = './door_mirror/training_cache/door_mirror'

import os
if not os.path.exists( storage_path ):
    os.makedirs( storage_path )

# dataset
import numpy as np
dataset = np.load( training_set_path )
camera_captured_rgb, screen_output_rgb = dataset['cameras'], dataset['screens']
print( f'training data [0] loaded from {training_set_path}' )

dataset_se = np.load( training_set_path_se )
camera_captured_rgb_se, screen_output_rgb_se = dataset_se['cameras'], dataset_se['screens']
print( f'training data [1] loaded from {training_set_path_se}' )

import gc
camera_captured_rgb = np.concatenate( [camera_captured_rgb, camera_captured_rgb_se], axis=0 )
camera_captured_rgb_se = None
gc.collect() # free memory or OOM
screen_output_rgb = np.concatenate( [screen_output_rgb, screen_output_rgb_se], axis=0 )
screen_output_rgb_se = None
gc.collect() # free memory or OOM
print( 'merged training dataset [0, 1]' )

# merge dataset [0, 1, 2]
dataset_rd = np.load( training_set_path_rd )
camera_captured_rgb_rd, screen_output_rgb_rd = dataset_rd['cameras'], dataset_rd['screens']
camera_captured_rgb = np.concatenate( [camera_captured_rgb, camera_captured_rgb_rd], axis=0 )
camera_captured_rgb_rd = None
gc.collect() # free memory or OOM
screen_output_rgb = np.concatenate( [screen_output_rgb, screen_output_rgb_rd], axis=0 )
screen_output_rgb_rd = None
gc.collect() # free memory or OOM
print( 'merged training dataset [0, 1, 2]' )

screen_output_rgb = ( screen_output_rgb - np.amin(screen_output_rgb) ) / ( np.amax(screen_output_rgb) - np.amin(screen_output_rgb) )
print( f'training set output normalized to range [{np.amin(screen_output_rgb)}, {np.amax(screen_output_rgb)}]' )

# prepare scaled outputs of different frequencies
from skimage.measure import block_reduce
def make_block_reduce( input_layers, dim=(2,2), mode=np.mean ):
    if len(input_layers.shape) == 4:
        dim = dim + (1,)
    stacked_layers = [ block_reduce( image, dim, mode ) for image in input_layers ]
    return np.asarray( stacked_layers, dtype='float32' )

screen_output_rgb_256 = make_block_reduce( screen_output_rgb )
print( 'screen_output_rgb_256 -- generated' )
screen_output_rgb_128 = make_block_reduce( screen_output_rgb_256 )
print( 'screen_output_channels_3_128 -- generated' )
screen_output_rgb_64 = make_block_reduce( screen_output_rgb_128 )
print( 'screen_output_channels_3_64 -- generated' )
screen_output_rgb_32 = make_block_reduce( screen_output_rgb_64 )
print( 'screen_output_channels_3_32 -- generated' )

def preprocess_neuralnetwork_input( array ):
    array = 2.0 * ( array - np.amin(array) ) / ( np.amax(array) - np.amin(array) + 1.0e-10 ) - 1.0
    return array
camera_captured_rgb = preprocess_neuralnetwork_input( camera_captured_rgb )
print( f'training set input normalized to range [{np.amin(camera_captured_rgb)}, {np.amax(camera_captured_rgb)}]' )

# load test dataset
e_dataset = np.load( testing_data_path )
test_camera_captured_images = e_dataset['cameras']
test_camera_captured_images = preprocess_neuralnetwork_input( test_camera_captured_images )
print( f'testing set input normalized to range [{np.amin(test_camera_captured_images)}, {np.amax(test_camera_captured_images)}]' )

iterations = 128 # weights and learning rate chaning ratio
epochs = 128 # training epochs per iteration
batch_size = 2 # training batch
model_path = f'{storage_path}/generator.model'

from mdcnn_door_mirror_model import build_model
from keras.optimizers import Adam
s_generator = build_model( (None, None, 3), output_channels=3 )
print( 'MDCNN model build' )

from keras.utils import multi_gpu_model
generator = multi_gpu_model( s_generator, gpus=2 )

import math
for iteration in range( iterations ):
    optimizer = Adam(lr=0.001*math.exp(-10.0*iteration/iterations)) # [1.0e-3 --> 4.5e-8]
    loss_weights = [ (2.0-2.0*iteration/iterations)**m for m in range( 5 ) ] #
    generator.compile( loss='mae',  optimizer=optimizer, metrics=['mae',], loss_weights=loss_weights )
    generator.fit( camera_captured_rgb, [screen_output_rgb, screen_output_rgb_256, screen_output_rgb_128, screen_output_rgb_64, screen_output_rgb_32], batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.125 )
    generator.save( f'{storage_path}/door_mirror_{iteration}.model')
    s_generator.save( model_path )


import imageio
prediction, *_ = generator.predict( test_camera_captured_images, batch_size=batch_size )
n, *_ = prediction.shape
for idx in range( n ):
    imageio.imsave( f'./prediction_{idx}.png', prediction[idx] )
    imageio.imsave( f'./camera_{idx}.png', test_camera_captured_images[idx] )


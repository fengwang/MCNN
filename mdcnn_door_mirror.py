dataset_path = '/data2/feng/door_mirror/dataset/set4/0_256_screens_cameras.npz_1280X7200_scaled_to_0_1.npz_flipped_cropped_0_1.npz'
dataset_path_se = '/data2/feng/door_mirror/dataset/set4/1_256_screens_cameras.npz_1280X7200_scaled_to_0_1.npz_flipped_cropped_0_1.npz'
dataset_path_rd = '/data2/feng/door_mirror/dataset/set4/2_256_screens_cameras.npz_1280X7200_scaled_to_0_1.npz_flipped_cropped_0_1.npz'
test_dataset_path = '/data2/feng/door_mirror/dataset/set4/3_256_screens_cameras.npz_1280X7200_scaled_to_0_1.npz_flipped_cropped_0_1.npz'
storage_path = './door_mirror/training_cache/door_mirror'

import os
if not os.path.exists( storage_path ):
    os.makedirs( storage_path )

# dataset
import numpy as np
rems = 0 # first few dataset are not used
dataset = np.load( dataset_path )
camera_captured_channel_3, screen_output_channel_3 = dataset['cameras'][rems:], dataset['screens'][rems:]
print( f'training data loaded from {dataset_path}' )

dataset_se = np.load( dataset_path_se )
camera_captured_channel_3_se, screen_output_channel_3_se = dataset_se['cameras'], dataset_se['screens']
print( f'training data loaded from {dataset_path_se}' )

import gc
# merge dataset [0, 1]
camera_captured_channel_3 = np.concatenate( [camera_captured_channel_3, camera_captured_channel_3_se], axis=0 )
camera_captured_channel_3_se = None
gc.collect() # free memory or OOM
screen_output_channel_3 = np.concatenate( [screen_output_channel_3, screen_output_channel_3_se], axis=0 )
screen_output_channel_3_se = None
gc.collect() # free memory or OOM
print( 'merged training dataset [0, 1]' )


# merge dataset [0, 1, 2]
dataset_rd = np.load( dataset_path_rd )
camera_captured_channel_3_rd, screen_output_channel_3_rd = dataset_rd['cameras'], dataset_rd['screens']
camera_captured_channel_3 = np.concatenate( [camera_captured_channel_3, camera_captured_channel_3_rd], axis=0 )
camera_captured_channel_3_rd = None
gc.collect() # free memory or OOM
screen_output_channel_3 = np.concatenate( [screen_output_channel_3, screen_output_channel_3_rd], axis=0 )
screen_output_channel_3_rd = None
gc.collect() # free memory or OOM
print( 'merged training dataset [0, 1, 2]' )

screen_output_channel_3 = ( screen_output_channel_3 - np.amin(screen_output_channel_3) ) / ( np.amax(screen_output_channel_3) - np.amin(screen_output_channel_3) )
print( 'screen_output_channel_3 -- generated' )

# prepare scaled inputs
from skimage.measure import block_reduce
def make_block_reduce( input_layers, dim=(2,2), mode=np.mean ):
    if len(input_layers.shape) == 4:
        dim = dim + (1,)
    stacked_layers = [ block_reduce( image, dim, mode ) for image in input_layers ]
    return np.asarray( stacked_layers, dtype='float32' )

screen_output_channel_3_256 = make_block_reduce( screen_output_channel_3 )
print( 'screen_output_channel_3_256 -- generated' )
screen_output_channel_3_128 = make_block_reduce( screen_output_channel_3_256 )
print( 'screen_output_channels_3_128 -- generated' )
screen_output_channel_3_64 = make_block_reduce( screen_output_channel_3_128 )
print( 'screen_output_channels_3_64 -- generated' )
screen_output_channel_3_32 = make_block_reduce( screen_output_channel_3_64 )
print( 'screen_output_channels_3_32 -- generated' )

def preprocess_neuralnetwork_input( array ):
    array = 2.0 * ( array - np.amin(array) ) / ( np.amax(array) - np.amin(array) + 1.0e-10 ) - 1.0
    return array

camera_captured_channel_3 = preprocess_neuralnetwork_input( camera_captured_channel_3 )

# load test dataset
e_dataset = np.load( test_dataset_path )
e_images = e_dataset['cameras']
test_camera_captured_images = e_images
test_camera_captured_images = preprocess_neuralnetwork_input( test_camera_captured_images )
print( f'test data 3 loaded' )

import keras.backend as K
from keras.models import load_model
import os.path
from mdcnn_door_mirror_model import build_model
from keras.optimizers import Adam
iterations = 32
epochs = 32
batch_size = 2
model_path = f'{storage_path}/generator.model'
s_generator = build_model( (None, None, 3), output_channels=3 )
print( 'MDCNN model build' )

from keras.utils import multi_gpu_model
generator = multi_gpu_model( s_generator, gpus=2 )

import math
for iteration in range( iterations ):
    optimizer = Adam(lr=0.001*math.np(-10.0*iteration/iterations))
    loss_weights = [ (2.0-2.0*iteration/iterations)**m for m in range( 5 ) ]
    generator.compile( loss='mae',  optimizer=optimizer, metrics=['mae',], loss_weights=loss_weights )
    generator.fit( camera_captured_channel_3, [screen_output_channel_3, screen_output_channel_3_256, screen_output_channel_3_128, screen_output_channel_3_64, screen_output_channel_3_32], batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.125 )
    generator.save( f'{storage_path}/door_mirror_{iteration}.model')
    s_generator.save( model_path )


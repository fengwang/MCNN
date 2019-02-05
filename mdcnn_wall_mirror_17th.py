# input is of shape ( 360X640 ) -- cameras
# output is of shape ( 720X1280 ) -- screens
dataset_path = '/data2/feng/wall_mirror/dataset/set4/0_256_screens_cameras.npz_1280X7200_scaled_to_0_1.npz_flipped_cropped_0_1.npz'
dataset_path_se = '/data2/feng/wall_mirror/dataset/set4/1_256_screens_cameras.npz_1280X7200_scaled_to_0_1.npz_flipped_cropped_0_1.npz'
test_dataset_path = '/data2/feng/wall_mirror/dataset/set4/2_256_screens_cameras.npz_1280X7200_scaled_to_0_1.npz_flipped_cropped_0_1.npz'
storage_path = './wall_mirror/training_cache/17th'

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def mkdir( directory ):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print( f'making directory {directory}' )

mkdir( storage_path )

# dataset
import numpy as np
import gc
rems = 0 # first few dataset are not used
dataset = np.load( dataset_path )
camera_captured_channel_3, screen_output_channel_3 = dataset['cameras'][rems:], dataset['screens'][rems:]
loaded_n, *_ = screen_output_channel_3.shape
total = loaded_n - rems
print( f'training data loaded from {dataset_path}' )

dataset_se = np.load( dataset_path_se )
camera_captured_channel_3_se, screen_output_channel_3_se = dataset_se['cameras'][rems:], dataset_se['screens'][rems:]
loaded_n, *_ = screen_output_channel_3_se.shape
total += loaded_n - rems
print( f'training data loaded from {dataset_path_se}' )

camera_captured_channel_3 = np.concatenate( [camera_captured_channel_3, camera_captured_channel_3_se], axis=0 )
camera_captured_channel_3_se = None
gc.collect() # free memory
screen_output_channel_3 = np.concatenate( [screen_output_channel_3, screen_output_channel_3_se], axis=0 )
screen_output_channel_3_se = None
gc.collect() # free memory
print( f'dataset merged' )


e_dataset = np.load( test_dataset_path )
print( f'test camera images {test_dataset_path} loaded' )
e_images = e_dataset['cameras']
test_camera_captured_images = e_images

### here
camera_captured_channel_3 = np.concatenate( [camera_captured_channel_3, e_images], axis=0 )
screen_output_channel_3 = np.concatenate( [screen_output_channel_3, e_dataset['screens']], axis=0 )
print( '768 dataset merged' )
### here

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
print( 'camera_captured_channel_3 -- generated' )

print( 'training set input preprocess_neuralnetwork_inputd' )

import imageio

#experimental data
def dump_all_images( parent_path, arrays ):
    n, row, col, *_ = arrays.shape
    for idx in range( n ):
        file_name = f'{parent_path}_{idx}.png'
        imageio.imsave( file_name, arrays[idx] )
        print( f'{file_name} dumped', end='\r' )
    print(' ')

dump_all_images( f'{storage_path}/test_camera_', test_camera_captured_images[0:8] )
print( 'test camera images dumped' )


test_camera_captured_images = preprocess_neuralnetwork_input( test_camera_captured_images )

print( 'test camera images preprocess_neuralnetwork_input done' )

test_screen_images = e_dataset['screens']
dump_all_images( f'{storage_path}/test_screens_', test_screen_images[0:8] )
print( 'test screen images dumped' )

print( f'test data loaded from {test_dataset_path}' )

import keras.backend as K
from keras.models import load_model
import os.path
from multidomain_wall_mirror_generator_17th import build_model
from keras.optimizers import Adam
optimizer = Adam()
iterations = 1024
epochs = 1
batch_size = 1 #single GPU case
generator_weights_path = f'{storage_path}/g_model.weights'
from keras.utils import multi_gpu_model
model_path = f'{storage_path}/generator.model'
if os.path.isfile( model_path ):
    generator = load_model( model_path )
else:
    generator = build_model( (None, None, 3), output_channels=3 )
    #generator = multi_gpu_model( generator, gpus=2 )


for iteration in range( iterations ):
    #loss_weights = [1.0, 8.0*(1.0 - iteration/iterations), 32.0*(1.0 - iteration/iterations), 128.0*(1.0 - iteration/iterations), 512.0*(1.0 - iteration/iterations)]
    loss_weights = [1.0, 2.0*(1.0 - iteration/iterations), 4.0*(1.0 - iteration/iterations), 8.0*(1.0 - iteration/iterations), 16.0*(1.0 - iteration/iterations)]
    generator.compile( loss=['mae', 'mae', 'mae', 'mae', 'mae'], loss_weights=loss_weights, optimizer=optimizer, metrics=['mae',] )

    generator.fit( camera_captured_channel_3, [screen_output_channel_3, screen_output_channel_3_256, screen_output_channel_3_128, screen_output_channel_3_64, screen_output_channel_3_32], batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.0 )
    #dump_all_images( f'{storage_path}/test_generated_{iteration}_', e_512 )
    generator.save( f'{storage_path}/test_cameras_{iteration}.model')
    generator.save( model_path )

    generator.save_weights( generator_weights_path )

    # dump
    directory = f'{storage_path}/dump_{iteration}/'
    mkdir( directory )
    print( f'trying to dump test cases for iteration:{iteration}' )
#!!! here
    K.clear_session()
    K.set_learning_phase(1)
    generator = load_model( model_path )
#!!! here
    e_512_all, e_256_all, e_128_all, e_64_all, e_32_all = generator.predict( test_camera_captured_images, batch_size=batch_size )
    dump_all_images( directory + 'dump_512_', e_512_all )
    #dump_all_images( directory + 'dump_256_', e_256_all )
    #dump_all_images( directory + 'dump_128_', e_128_all )
    #dump_all_images( directory + 'dump_64_', e_64_all )
    #dump_all_images( directory + 'dump_32_', e_32_all )
    e_512_all, e_256_all, e_128_all, e_64_all, e_32_all = None, None, None, None, None

    print( f'dumped test cases for iteration:{iteration}' )

    gc.collect() # free memory


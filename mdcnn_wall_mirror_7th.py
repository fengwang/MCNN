# input is of shape ( 360X640 ) -- cameras
# output is of shape ( 720X1280 ) -- screens
dataset_path = '/data2/feng/wall_mirror/dataset/set4/0_256_screens_cameras.npz_1280X7200_scaled_to_0_1.npz_cropped_flipped_upscaled_teshape.npz'
test_dataset_path = '/data2/feng/wall_mirror/dataset/set4/1_256_screens_cameras.npz_1280X7200_scaled_to_0_1.npz_cropped_flipped_upscaled_teshape.npz'

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from keras.layers import Input
input_3 = Input( (None, None, 3) )


# dataset
import numpy as np
rems = 0 # first few dataset are not used
dataset = np.load( dataset_path )
camera_captured_channel_3, screen_output_channel_3 = dataset['cameras'][rems:], dataset['screens'][rems:]
loaded_n, *_ = screen_output_channel_3.shape
total = loaded_n - rems
print( f'training data loaded from {dataset_path}' )

screen_output_channel_3 = ( screen_output_channel_3 - np.amin(screen_output_channel_3) ) / ( np.amax(screen_output_channel_3) - np.amin(screen_output_channel_3) )
print( 'screen_output_channel_3 -- generated' )
print( 'camera_captured_channel_3 -- generated' )

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


def preprocess_neuralnetwork_input( array ):
    array = 2.0 * ( array - np.amin(array) ) / ( np.amax(array) - np.amin(array) + 1.0e-10 ) - 1.0
    return array

camera_captured_channel_3 = preprocess_neuralnetwork_input( camera_captured_channel_3 )

print( 'training set input preprocess_neuralnetwork_inputd' )

import imageio

#experimental data
def dump_all_images( parent_path, arrays ):
    n, row, col, *_ = arrays.shape
    for idx in range( n ):
        file_name = f'{parent_path}_{idx}.png'
        imageio.imsave( file_name, arrays[idx] )
        print( f'{file_name} dumped' )

e_dataset = np.load( test_dataset_path )
print( f'test camera images {test_dataset_path} loaded' )
e_images = e_dataset['cameras']
test_camera_captured_images = e_images
dump_all_images( './wall_mirror/test_camera_', test_camera_captured_images[0:8] )
print( 'test camera images dumped' )

test_camera_captured_images = preprocess_neuralnetwork_input( test_camera_captured_images )
test_camera_captured_data = test_camera_captured_images[0:1]

print( 'test camera images preprocess_neuralnetwork_input done' )

test_screen_images = e_dataset['screens']
dump_all_images( './wall_mirror/test_screens_', test_screen_images[0:8] )
print( 'test screen images dumped' )

print( f'test data loaded from {test_dataset_path}' )


import keras.backend as K
from keras.models import load_model
import os.path

import os
def mkdir( directory ):
    if not os.path.exists(directory):
        os.makedirs(directory)

from multidomain_wall_mirror_generator_7th import build_model
from keras.optimizers import Adam
optimizer = Adam()
iterations = 1024
batch_size = 1
generator_weights_path = f'./wall_mirror/g_model.weights'
generator = build_model( (None, None, 3), output_channels=3 )
for iteration in range( iterations ):

    #if os.path.isfile(generator_weights_path):
    #    generator.load_weights( generator_weights_path )
    #    print( f'generator weights loaded from {generator_weights_path}' )

    loss_weights = [1.0, (1.0 - iteration/iterations), (1.0 - iteration/iterations) ]
    generator.compile( loss=['mae', 'mae', 'mae'], loss_weights=loss_weights, optimizer=optimizer )

    generator.fit( camera_captured_channel_3, [screen_output_channel_3, screen_output_channel_3_256, screen_output_channel_3_128], batch_size=batch_size, epochs=1, verbose=0, validation_split=0.0 )
    e_512, e_256, e_128 = generator.predict( test_camera_captured_data, batch_size=batch_size )
    dump_all_images( f'./wall_mirror/test_generated_{iteration}_', e_512 )
    generator.save( f'./wall_mirror/test_cameras_{iteration}.model')

    generator.save_weights( generator_weights_path )

    # dump
    directory = f'./wall_mirror/dump_{iteration}/'
    mkdir( directory )
    print( f'trying to dump test cases for iteration:{iteration}' )
    if ((iteration+1) % 8) == 0:
        e_512_all, *_ = generator.predict( test_camera_captured_images, batch_size=batch_size )
        dump_all_images( directory + 'dump_', e_512_all )
    else:
        e_512_all, *_ = generator.predict( test_camera_captured_images[0:8], batch_size=batch_size )
        dump_all_images( directory + 'dump_', e_512_all )

    print( f'dumped test cases for iteration:{iteration}' )

    #K.clear_session() #<<-- clear


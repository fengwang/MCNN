# input is of shape ( 360X640 ) -- cameras
# output is of shape ( 720X1280 ) -- screens
#load more data
# prepare scaled inputs
import numpy as np
from skimage.measure import block_reduce
def make_block_reduce( input_layers, dim=(2,2), mode=np.mean ):
    if len(input_layers.shape) == 4:
        dim = dim + (1,)
    stacked_layers = [ block_reduce( image, dim, mode ) for image in input_layers ]
    return np.asarray( stacked_layers, dtype='float32' )

def preprocess_neuralnetwork_input( array ):
    array = 2.0 * ( array - np.amin(array) ) / ( np.amax(array) - np.amin(array) + 1.0e-10 ) - 1.0
    return np.asarray( array, dtype='float32' )

dataset_count = 7
dataset_paths = [f'/data2/feng/wall_mirror/dataset/set4/{idx}_256_screens_cameras.npz_1280X7200_scaled_to_0_1.npz_flipped_cropped_0_1.npz' for idx in range( dataset_count ) ]
# load data
dataset_loaded = []
for path in dataset_paths:
    dataset = np.load( path )
    camera_captured_channel_3, screen_output_channel_3 = dataset['cameras'], dataset['screens']
    camera_captured_channel_3 = preprocess_neuralnetwork_input( camera_captured_channel_3 )
    screen_output_channel_3_256 = make_block_reduce( screen_output_channel_3 )
    screen_output_channel_3_128 = make_block_reduce( screen_output_channel_3_256 )
    screen_output_channel_3_64  = make_block_reduce( screen_output_channel_3_128 )
    screen_output_channel_3_32  = make_block_reduce( screen_output_channel_3_64 )
    dataset_loaded.append( [ camera_captured_channel_3,
                           [screen_output_channel_3, screen_output_channel_3_256, screen_output_channel_3_128, screen_output_channel_3_64, screen_output_channel_3_32] ] )
    print (f'finished loading data from {path}' )

import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

def mkdir( directory ):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print( f'making directory {directory}' )

storage_path = '/data2/feng/wall_mirror/training_cache/9th'
mkdir( storage_path )

import imageio

def dump_all_images( parent_path, arrays ):
    n, row, col, *_ = arrays.shape
    for idx in range( n ):
        file_name = f'{parent_path}_{idx}.png'
        imageio.imsave( file_name, arrays[idx] )
        print( f'{file_name} dumped', end='\r' )
    print(f'all images from {parent_path} dumped')

import keras.backend as K
from keras.models import load_model
import os.path
from multidomain_wall_mirror_generator_8th import build_model
from keras.optimizers import Adam
import gc
optimizer = Adam()
iterations = 1024
batch_size = 2
generator = build_model( (None, None, 3), output_channels=3 )
# try load old data here
from keras.utils import multi_gpu_model
m_generator = multi_gpu_model( generator, gpus=2 )
m_generator.compile(loss='mae', optimizer=optimizer)
for iteration in range( iterations ):
    # we are lack of memory, doing piece-wise training here
    for training_idx in range( dataset_count-1 ):
        m_generator.fit( dataset_loaded[training_idx+1][0], dataset_loaded[training_idx+1][1], batch_size=batch_size, epochs=1, verbose=0, validation_split=0 )
    generator.save( f'{storage_path}/model_generator.model' )
    m_generator.save( f'{storage_path}/model_m_generator_{iteration}.model' )
    directory = f'{storage_path}/dump_{iteration}/'
    mkdir( directory )
    if iteration % 8 == 0:
        e_512, *_ = m_generator.predict( dataset_loaded[0][0], batch_size=batch_size )
    else:
        e_512, *_ = m_generator.predict( dataset_loaded[0][0][0:32], batch_size=batch_size )
    dump_all_images( directory + 'dump_', e_512 )
    e_512, _ = None, None
    gc.collect()
    #K.clear_session() #<<-- clear


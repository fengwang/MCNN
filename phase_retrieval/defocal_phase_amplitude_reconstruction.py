import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--generator_input_channels', type=int ) # images to use
parser.add_argument('--total', type=int, default=2048 ) # samples in the training set to use
parser.add_argument('--batch_size', type=int, default=4 )
parser.add_argument('--iterations', type=int, default=256 )
parser.add_argument('--sampling_interval', type=int, default=8 )
parser.add_argument('--dim', type=int, default=512 ) # the image size used during training
parser.add_argument('--gpus', type=int, default=2 )

args                        = parser.parse_args()
generator_input_channels    = args.generator_input_channels
total                       = args.total
batch_size                  = args.batch_size
iterations                  = args.iterations
sampling_interval           = args.sampling_interval
dim                         = args.dim
gpus                        = args.gpus

generator_output_channels = 2 # phase and amplitude

from datetime import datetime
cache_path = f'./training_cache/defocal_reconstruction_gi_input_channels_{generator_input_channels}_{datetime.now()}'

import os
if not os.path.isdir(cache_path):
    os.mkdir( cache_path )

generator_input_channels_offset = (51-generator_input_channels)>>1

def extract_input( array, start_idx=0 ):
    print( f'extracting stride array of shape {array.shape} at index {start_idx} with generator_input_channels {generator_input_channels}', end='\r' )
    if len( array.shape ) == 3:
        return array[:,:,generator_input_channels_offset:generator_input_channels_offset+generator_input_channels]
    if len( array.shape ) == 4:
        return array[:,:,:,generator_input_channels_offset:generator_input_channels_offset+generator_input_channels]
    assert False, 'Error: only for numpy arrays of 3D or 4D'

from multidomain_generator import build_model
generator = build_model( (None, None, generator_input_channels), output_channels=generator_output_channels )

if gpus > 1 :
    from keras.utils import multi_gpu_model
    generator = multi_gpu_model( generator, gpus=gpus )

generator.compile( loss='mae', optimizer='adam', metrics=['accuracy'] )

def central_crop( input_array, shape ):
    _, r, c, _ = input_array.shape
    _r, _c = shape
    offset_r, offset_c = ((r-_r)>>1), ((c-_c)>>1)
    return input_array[:, offset_r:offset_r+_r, offset_c:offset_c+_c, :]

import numpy as np
def norm( input ):
    input = (input-np.mean(input))/(np.std(input)+1.0e-10)
    return input.astype('float32')

def uniform( input ):
    input = (input-np.amin(input))/(np.amax(input)-np.amin(input)+1.0e-10)
    return input.astype('float32')

# load exp
exp_data_ = np.load('/raid/feng/experimental_data/defocused.51.depth.image.npz')
exp_data_ = exp_data_['image'] # (51, 2160, 2560)
exp_data_ = exp_data_[:, 56+896+64:56+896+64+640, 256+896+64:256+896+64+640] # crop
exp_data_ = np.moveaxis(exp_data_, 0, -1) # channel last
exp_data = extract_input( exp_data_ ) # extract data
exp_data = exp_data.reshape( (1, 640, 640, generator_input_channels) )
exp_data = norm( exp_data )

# load training data
import glob
training_records = glob.glob( '/raid/feng/simulated_data/*.npy' )
from random import shuffle
shuffle(training_records)

if total > len(training_records):
    total = len(training_records)

import gc
def transform( record_path, generator_input_channels=generator_input_channels ):
    data = np.load( record_path )
    print( f'loading records from {record_path}, with shape {data.shape}', end='\r' )
    o_2 = data[:, :, :generator_output_channels]
    o_2 = o_2.reshape( (1,)+o_2.shape )
    o_2 = central_crop( o_2, (dim, dim) )

    i_4 = extract_input( data, generator_output_channels ) # extract data
    i_4 = i_4.reshape( (1,)+i_4.shape )
    i_4 = central_crop( i_4, (dim, dim) )
    data = None
    gc.collect()
    return i_4, o_2

generator_input_channel_51 = np.zeros( (total, dim, dim, generator_input_channels), dtype='float32' )
generator_output_channels_2 = np.zeros( (total, dim, dim, generator_output_channels), dtype='float32' )

for idx in range( total ):
    generator_input_channel_51[idx], generator_output_channels_2[idx] = transform( training_records[idx] )
generator_input_channel_51 = norm( generator_input_channel_51 )
i_1 = generator_input_channel_51

generator_output_channels_2 = uniform( generator_output_channels_2 )
from skimage.measure import block_reduce
o_1 = generator_output_channels_2
o_2 = block_reduce(o_1, block_size=(1, 2, 2, 1), func=np.mean)
o_3 = block_reduce(o_2, block_size=(1, 2, 2, 1), func=np.mean)

import keras.backend as K
import imageio

for iteration in range( iterations ):

    generator_model_path = f'{cache_path}/g_model_{iteration}.h5'

    for idx in range( int(total/batch_size) ):
        start = idx * batch_size
        end = start + batch_size
        input_512 = i_1[start:end]
        output_512 = o_1[start:end]
        output_256 = o_2[start:end]
        output_128 = o_3[start:end]

        g_loss = generator.train_on_batch( input_512, [output_512,output_256,output_128] )
        print( f'iteration: {iteration}/{iterations}: {g_loss}' )

        if idx % sampling_interval == 0:
            K.set_learning_phase( 1 )
            e_512, *_ = generator.predict( exp_data )
            image_path = f'{cache_path}/_{iteration}_{idx}_a.png'
            imageio.imsave( image_path, np.squeeze( e_512[:, :, :, 0] ) )
            a_image_path = f'{cache_path}/_{iteration}_{idx}_p.png'
            imageio.imsave( a_image_path, np.squeeze( e_512[:, :, :, 1] ) )

    generator.save( generator_model_path )
    gc.collect()


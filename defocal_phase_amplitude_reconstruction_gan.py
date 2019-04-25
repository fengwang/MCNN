import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_channels', type=int )
args = parser.parse_args()
generator_input_channels = args.input_channels
input_channels = args.input_channels

generator_output_channels = 2
generator_input_channels = generator_input_channels

from datetime import datetime
cache_path = f'./training_cache/defocused_reconstruction_gi_{generator_input_channels}_input_channels_{generator_input_channels}_{datetime.now()}'

import os
if not os.path.isdir(cache_path):
    os.mkdir( cache_path )

input_channels_offset = (51-generator_input_channels)>>1

def extract_stride_input( array, start_idx=0 ):
    print( f'extracting stride array of shape {array.shape} at index {start_idx} with input_channels {generator_input_channels}', end='\r' )
    if len( array.shape ) == 3:
        return array[:,:,input_channels_offset:input_channels_offset+input_channels]
    if len( array.shape ) == 4:
        return array[:,:,:,input_channels_offset:input_channels_offset+input_channels]
    assert False, 'only stride extract for numpy arrays of 3D or 4D'

from keras.layers import Input
input_51 = Input( (None, None, generator_input_channels) )

from multidomain_generator import build_model
generator = build_model( (None, None, generator_input_channels), output_channels=generator_output_channels )
from keras.utils import multi_gpu_model
generator = multi_gpu_model( generator, gpus=2 )
o_512, o_256, o_128 = generator( input_51 )

discriminator_input_channels = generator_output_channels
discriminator_output_channels = 2

from discriminator_se import build_discriminator
discriminator = build_discriminator( (None, None, discriminator_input_channels), output_channels = discriminator_output_channels )
discriminator = multi_gpu_model( discriminator, gpus=2 )

from keras.optimizers import Adam
optimizer = Adam( 0.0002, 0.5 )
discriminator.compile( loss='mse', optimizer=optimizer, metrics=['accuracy'] )
discriminator.Trainable = False
o_32, o_16, o_8 = discriminator( o_512 )

from keras.models import Model
gan = Model( inputs=input_51, outputs=[o_512, o_256, o_128, o_32, o_16, o_8] )
gan = multi_gpu_model( gan, gpus=2 )
gan.compile( loss=['mae', 'mae', 'mae', 'mse', 'mse', 'mse'], loss_weights=[100, 100, 100, 1, 1, 1], optimizer=optimizer )

# dataset
import numpy as np

def trim_to( input_array, shape ):
    _, r, c, _ = input_array.shape
    _r, _c = shape
    offset_r, offset_c = ((r-_r)>>1), ((c-_c)>>1)
    return input_array[:, offset_r:offset_r+_r, offset_c:offset_c+_c, :]

def report( array, message=None ):
    if message:
        print( f'{message}' )
    mx = np.amax( array )
    me = np.mean( array )
    mn = np.amin( array )
    st = np.std( array )
    print( f'max-{mx}; mean-{me}; min-{mn}; std-{st}' )

total = int( 51 / generator_input_channels ) * 320

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
exp_data = extract_stride_input( exp_data_ ) # extract data
exp_data = exp_data.reshape( (1, 640, 640, generator_input_channels) )
exp_data = norm( exp_data )
report( exp_data, 'exp_data' )

# load training data
import glob
training_records = glob.glob( '/raid/feng/simulated_data/jellyfish_defocus_51_noisy/*.npy' )
dim = 512
from random import shuffle
shuffle(training_records)

if total > len(training_records):
    total = len(training_records)

import gc
def transform( record_path, input_channels=generator_input_channels ):
    data = np.load( record_path )
    print( f'loading records from {record_path}, with shape {data.shape}', end='\r' )
    o_2 = data[:, :, :generator_output_channels]
    o_2 = o_2.reshape( (1,)+o_2.shape )
    o_2 = trim_to( o_2, (dim, dim) )

    *_, chs = data.shape
    #i_4 = data[:,:,generator_output_channels::2]
    i_4 = extract_stride_input( data, generator_output_channels ) # extract data
    i_4 = i_4.reshape( (1,)+i_4.shape )
    i_4 = trim_to( i_4, (dim, dim) )
    data = None
    gc.collect()
    return i_4, o_2

imgs_to_train = total
generator_input_channel_51 = np.zeros( (imgs_to_train, dim, dim, generator_input_channels), dtype='float32' )
generator_output_channels_2 = np.zeros( (imgs_to_train, dim, dim, generator_output_channels), dtype='float32' )

for idx in range( imgs_to_train ):
    generator_input_channel_51[idx], generator_output_channels_2[idx] = transform( training_records[idx] )

generator_input_channel_51 = norm( generator_input_channel_51 )
generator_output_channels_2 = uniform( generator_output_channels_2 )

report( generator_input_channel_51, 'generator_input_channel_51' )
report( generator_output_channels_2, 'generator_output_channels_2' )

from skimage.measure import block_reduce
o_1 = generator_output_channels_2
o_2 = block_reduce(o_1, block_size=(1, 2, 2, 1), func=np.mean)
o_3 = block_reduce(o_2, block_size=(1, 2, 2, 1), func=np.mean)
#o_4 = block_reduce(o_3, block_size=(1, 2, 2, 1), func=np.mean)

report( o_1, 'o_1' )
report( o_2, 'o_2' )
report( o_3, 'o_3' )

input_512 = o_1
input_256 = o_2
input_128 = o_3

def normalize( array ):
    array = ( array - np.mean(array) ) / ( np.std(array) + 1.0e-10 )
    return array

output_4 = generator_input_channel_51

experimental_data_channel_51 = exp_data
report( experimental_data_channel_51, 'experimental_data_channel_51' )

batch_size = 4
channels = discriminator_output_channels

dims = [dim//16, dim//32, dim//64]

valid_32 = np.ones( (batch_size, dims[0], dims[0], channels) )
valid_16 = np.ones( (batch_size, dims[1], dims[1], channels) )
valid_8 = np.ones( (batch_size, dims[2], dims[2], channels) )

fake_32 = np.zeros( (batch_size, dims[0], dims[0], channels) )
fake_16 = np.zeros( (batch_size, dims[1], dims[1], channels) )
fake_8 = np.zeros( (batch_size, dims[2], dims[2], channels) )

iterations = 256
sampling_interval = 8
import keras.backend as K
from keras.models import load_model
import os.path
import imageio

for iteration in range( iterations ):

    generator_model_path = f'{cache_path}/g_model_{iteration}.h5'

    discriminator_model_path = f'{cache_path}/d_model_{iteration}.h5'

    gan_model_path = f'{cache_path}/gan_model_{iteration}.h5'

    for idx in range( int(total/batch_size) ):
        start = idx * batch_size
        end = start + batch_size
        input_51 = output_4[start:end, :, :, :]
        output_512 = input_512[start:end, :, :, :]
        output_256 = input_256[start:end, :, :, :]
        output_128 = input_128[start:end, :, :, :]

        fake_512, fake_256, fake_128 = generator.predict( input_51 )
        d_loss_real = discriminator.train_on_batch( output_512, [valid_32, valid_16, valid_8] )
        d_loss_fake = discriminator.train_on_batch( fake_512, [fake_32, fake_16, fake_8] )
        d_loss = np.add( d_loss_real, d_loss_fake ) * 0.5

        gan_loss = gan.train_on_batch( input_51, [output_512, output_256, output_128, valid_32, valid_16, valid_8] )
        print( f'iteration: {iteration}/{iterations}: d_loss:{d_loss} and gan_loss:{gan_loss}' )

        if idx % sampling_interval == 0:
            e_512, e_256, e_128 = generator.predict( experimental_data_channel_51 )
            image_path = f'{cache_path}/_{iteration}_{idx}_a.png'
            imageio.imsave( image_path, np.squeeze( e_512[:, :, :, 0] ) )
            a_image_path = f'{cache_path}/_{iteration}_{idx}_p.png'
            imageio.imsave( a_image_path, np.squeeze( e_512[:, :, :, 1] ) )

    generator.save( generator_model_path )
    discriminator.save( discriminator_model_path )
    gan.save( gan_model_path )
    gc.collect()


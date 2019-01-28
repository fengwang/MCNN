dataset_path = '/data2/feng/wall_mirror/0_256_screens_cameras.npz_1280X7200_scaled_to_0_1.npz'
test_dataset_path = '/data2/feng/wall_mirror/1_256_screens_cameras.npz_1280X7200_scaled_to_0_1.npz'

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from keras.layers import Input
input_3 = Input( (None, None, 3) )

from multidomain_wall_mirror_generator import build_model
generator = build_model( (None, None, 3), output_channels=3 )
o_512, o_256, o_128 = generator( input_3 )

from discriminator import build_discriminator
discriminator = build_discriminator( (None, None, 3), output_channels = 1 )
from keras.optimizers import Adam
optimizer = Adam( 0.0002, 0.5 )
discriminator.compile( loss='mse', optimizer=optimizer, metrics=['accuracy'] )
discriminator.Trainable = False
o_32, o_16, o_8 = discriminator( o_512 )

from keras.models import Model
gan = Model( inputs=input_3, outputs=[o_512, o_256, o_128, o_32, o_16, o_8] )
gan.compile( loss=['mae', 'mae', 'mae', 'mse', 'mse', 'mse'], loss_weights=[100, 50, 10, 1, 2, 4], optimizer=optimizer )

# dataset
import numpy as np
rems = 0 # first few dataset are not used
dataset = np.load( dataset_path )
camera_captured_channel_3, screen_output_channel_3 = dataset['cameras'][rems:], dataset['screens'][rems:]
loaded_n, *_ = screen_output_channel_3.shape
total = loaded_n - rems

camera_input_average = np.sum( camera_captured_channel_3, axis=0 ) / loaded_n

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

input_1_256 = make_block_reduce( screen_output_channel_3 )
print( 'input_1_256 -- generated' )
input_1_128 = make_block_reduce( input_1_256 )
print( 'input_1_128 -- generated' )


#def preprocess_neuralnetwork_input( array ):
#    array = 2.0 * ( array - np.amin(array) ) / ( np.amax(array) - np.amin(array) + 1.0e-10 ) - 1.0
#    return array

def preprocess_neuralnetwork_input( arrays ):
    #nonlocal camera_input_average
    for array in arrays:
        array -=  camera_input_average
    return arrays

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
dump_all_images( './wall_mirror/test_camera_', test_camera_captured_images )
print( 'test camera images dumped' )

test_camera_captured_images = preprocess_neuralnetwork_input( test_camera_captured_images )
test_camera_captured_data = test_camera_captured_images[0:1]

print( 'test camera images preprocess_neuralnetwork_input done' )

test_screen_images = e_dataset['screens']
dump_all_images( './wall_mirror/test_screens_', test_screen_images )
print( 'test screen images dumped' )

print( f'test data loaded from {test_dataset_path}' )

batch_size = 1
#valid_32 = np.ones( (batch_size, 32, 32, 1) )
#valid_16 = np.ones( (batch_size, 16, 16, 1) )
#valid_8 = np.ones( (batch_size, 8, 8, 1) )
#fake_32 = np.zeros( (batch_size, 32, 32, 1) )
#fake_16 = np.zeros( (batch_size, 16, 16, 1) )
#fake_8 = np.zeros( (batch_size, 8, 8, 1) )

valid_32 = np.ones( (batch_size, 45, 80, 1) )
valid_16 = np.ones( (batch_size, 22, 40, 1) )
valid_8 = np.ones( (batch_size, 11, 20, 1) )
fake_32 = np.zeros( (batch_size, 45, 80, 1) )
fake_16 = np.zeros( (batch_size, 22, 40, 1) )
fake_8 = np.zeros( (batch_size, 11, 20, 1) )

iterations = 1024

import keras.backend as K
from keras.models import load_model
import os.path

import os
def mkdir( directory ):
    if not os.path.exists(directory):
        os.makedirs(directory)

for iteration in range( iterations ):

    generator_model_path = f'./wall_mirror/g_model.h5'
    if os.path.isfile(generator_model_path):
        generator = load_model( generator_model_path )

    discriminator_model_path = f'./wall_mirror/d_model.h5'
    if os.path.isfile(discriminator_model_path):
        discriminator = load_model( discriminator_model_path )

    gan_model_path = f'./wall_mirror/gan_model.h5'
    if os.path.isfile(gan_model_path):
        gan = load_model( gan_model_path )

    for idx in range( int(total/batch_size) ):
        start = idx * batch_size
        end = start + batch_size
        input_3 = camera_captured_channel_3[start:end, :, :, :]
        output_512 = screen_output_channel_3[start:end, :, :, :]
        output_256 = input_1_256[start:end, :, :, :]
        output_128 = input_1_128[start:end, :, :, :]

        d_loss_real = discriminator.train_on_batch( output_512, [valid_32, valid_16, valid_8] )
        fake_512, fake_256, fake_128 = generator.predict( input_3 )
        d_loss_fake = discriminator.train_on_batch( fake_512, [fake_32, fake_16, fake_8] )
        d_loss = np.add( d_loss_real, d_loss_fake ) * 0.5

        gan_loss = gan.train_on_batch( input_3, [output_512, output_256, output_128, valid_32, valid_16, valid_8] )
        print( f'iteration: {iteration}/{iterations}: d_loss:{d_loss} and gan_loss:{gan_loss}' )

        e_512, e_256, e_128 = generator.predict( test_camera_captured_data )
        dump_all_images( f'./wall_mirror/test_generated_{iteration}_{idx}_', e_512 )

        if idx % 32 == 0:
            generator.save( f'./wall_mirror/test_cameras_{iteration}_{idx}.model')

    generator.save( generator_model_path )
    discriminator.save( discriminator_model_path )
    gan.save( gan_model_path )

    # dump all
    directory = f'./wall_mirror/dump_{iteration}/'
    mkdir( directory )
    print( f'trying to dump test cases for iteration:{iteration}' )
    e_512_all, *_ = generator.predict( test_camera_captured_images, batch_size=batch_size )
    dump_all_images( directory + 'dump_', e_512_all )
    print( f'dumped test cases for iteration:{iteration}' )

    K.clear_session() #<<-- clear


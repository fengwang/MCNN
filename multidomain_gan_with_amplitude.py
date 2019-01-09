import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from keras.layers import Input
input_33 = Input( (None, None, 33) )

from multidomain_generator import build_model
generator = build_model( (None, None, 33), output_channels=2 )
o_512, o_256, o_128 = generator( input_33 )

from discriminator import build_discriminator
discriminator = build_discriminator( (None, None, 2), output_channels = 2 )
from keras.optimizers import Adam
optimizer = Adam( 0.0002, 0.5 )
discriminator.compile( loss='mse', optimizer=optimizer, metrics=['accuracy'] )
discriminator.Trainable = False
o_32, o_16, o_8 = discriminator( o_512 )

from keras.models import Model
gan = Model( inputs=input_33, outputs=[o_512, o_256, o_128, o_32, o_16, o_8] )
gan.compile( loss=['mae', 'mae', 'mae', 'mse', 'mse', 'mse'], loss_weights=[100, 50, 10, 1, 2, 4], optimizer=optimizer )

# dataset
import numpy as np
total = 384 # total images, 512, 384
dataset_path =f'/data/cache/astigmatism_hela_rand_{total}-input_1-output_33-with_amplitude.npz'
dataset = np.load( dataset_path )
#input_1_512, output_33 = dataset['input_1'], dataset['output_33']
input_1_512, input_2_512, output_33 = dataset['input_1'], dataset['input_2'], dataset['output_33']
input_1_512 = ( input_1_512 - np.amin(input_1_512) ) / ( np.amax(input_1_512) - np.amin(input_1_512) )
input_2_512 = ( input_2_512 - np.amin(input_2_512) ) / ( np.amax(input_2_512) - np.amin(input_2_512) )
print( 'input_1_512 -- generated' )
print( 'output_33 -- generated' )

# prepare scaled inputs
from skimage.measure import block_reduce
def make_block_reduce( input_layers, dim=(2,2), mode=np.mean ):
    if len(input_layers.shape) == 4:
        dim = dim + (1,)
    stacked_layers = [ block_reduce( image, dim, mode ) for image in input_layers ]
    return np.asarray( stacked_layers, dtype='float32' )

input_1_256 = make_block_reduce( input_1_512 )
input_2_256 = make_block_reduce( input_2_512 )
print( 'input_1_256 -- generated' )
input_1_128 = make_block_reduce( input_1_256 )
input_2_128 = make_block_reduce( input_2_256 )
print( 'input_1_128 -- generated' )

input_512 = np.concatenate( [input_1_512, input_2_512], -1 )
#input_256
input_256 = np.concatenate( [input_1_256, input_2_256], -1 )
#input_128
input_128 = np.concatenate( [input_1_128, input_2_128], -1 )


def normalize( array ):
    array = ( array - np.mean(array) ) / ( np.std(array) + 1.0e-10 )
    return array

#normalize input for each image
for idx in range( total ):
    for jdx in range( 33 ):
        output_33[idx, :, :, jdx] = normalize( output_33[idx, :, :, jdx] )



#experimental data
e_path = '/data/experimental/Feng2018/HeLa_cell_astig_series/cropped_normalized_1536_1546-images.npz'
e_dataset = np.load( e_path )
e_images = e_dataset['images']
e_data_512_512_3 = e_images.reshape( (1,)+e_images.shape )

# fix:
# trim size to 1024
e_n, e_r, e_c, e_h = e_data_512_512_3.shape
e_size = 1024
offsets = [ (x-e_size)>>1 for x in (e_r, e_c) ]
e_data_512_512_3 = e_data_512_512_3[:, offsets[0]:offsets[0]+e_size:, offsets[1]:offsets[1]+e_size, : ]
for _n in range( e_n ):
    for _h in range( e_h ):
        e_data_512_512_3[_n, :, :, _h] = normalize( e_data_512_512_3[_n, :, :, _h] )
#end of fix

batch_size = 2
channels = 2
valid_32 = np.ones( (batch_size, 32, 32, channels) )
valid_16 = np.ones( (batch_size, 16, 16, channels) )
valid_8 = np.ones( (batch_size, 8, 8, channels) )

fake_32 = np.zeros( (batch_size, 32, 32, channels) )
fake_16 = np.zeros( (batch_size, 16, 16, channels) )
fake_8 = np.zeros( (batch_size, 8, 8, channels) )

iterations = 1024

import keras.backend as K
from keras.models import load_model
import os.path
import imageio

for iteration in range( iterations ):

    generator_model_path = f'./astig_33_gan/g_model.h5'
    if os.path.isfile(generator_model_path):
        generator = load_model( generator_model_path )

    discriminator_model_path = f'./astig_33_gan/d_model.h5'
    if os.path.isfile(discriminator_model_path):
        discriminator = load_model( discriminator_model_path )

    gan_model_path = f'./astig_33_gan/gan_model.h5'
    if os.path.isfile(gan_model_path):
        gan = load_model( gan_model_path )

    for idx in range( int(total/batch_size) ):
        start = idx * batch_size
        end = start + batch_size
        input_33 = output_33[start:end, :, :, :]
        output_512 = input_1_512[start:end, :, :, :]
        output_256 = input_1_256[start:end, :, :, :]
        output_128 = input_1_128[start:end, :, :, :]

        fake_512, fake_256, fake_128 = generator.predict( input_33 )
        d_loss_real = discriminator.train_on_batch( output_512, [valid_32, valid_16, valid_8] )
        d_loss_fake = discriminator.train_on_batch( fake_512, [valid_32, valid_16, valid_8] )
        d_loss = np.add( d_loss_real, d_loss_fake ) * 0.5

        gan_loss = gan.train_on_batch( input_33, [output_512, output_256, output_128, valid_32, valid_16, valid_8] )
        print( f'iteration: {iteration}/{iterations}: d_loss:{d_loss} and gan_loss:{gan_loss}' )

        e_512, e_256, e_128 = generator.predict( e_data_512_512_3 )
        image_path = f'./astig_33_gan/_{iteration}_{idx}_p.png'
        imageio.imsave( image_path, np.squeeze( e_512[:, :, :, 0] ) )
        a_image_path = f'./astig_33_gan/_{iteration}_{idx}_a.png'
        imageio.imsave( a_image_path, np.squeeze( e_512[:, :, :, 1] ) )
    generator.save( generator_model_path )
    discriminator.save( discriminator_model_path )
    gan.save( gan_model_path )

    K.clear_session() #<<-- clear


















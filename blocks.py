from keras import regularizers
from keras.layers import Input
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.activations import sigmoid
from keras.models import Model
from keras.layers import BatchNormalization
from keras.layers import concatenate
from keras.layers import LeakyReLU
from keras.layers.merge import add

def make_activation( input_layer ):
    return LeakyReLU(alpha=0.2)(BatchNormalization(momentum=0.8)(input_layer))

def make_block( input_layer, channels, kernel_size=(3,3) ):
    x = input_layer
    x = Conv2DTranspose( channels, kernel_size=kernel_size, activation='linear', strides=1, padding='valid' )( x )
    x = make_activation( x )
    x = Conv2D( channels, kernel_size=kernel_size, activation='linear', strides=1, padding='valid' )( x )
    x = make_activation( x )
    return x

def make_output_block( input_layer, twin_channels, kernel_size, output_activation ):
    channels, output_channels = twin_channels
    x = input_layer
    x = Conv2DTranspose( channels, kernel_size=kernel_size, activation='linear', strides=1, padding='valid' )( x )
    x = make_activation( x )
    x = Conv2D( output_channels, kernel_size=kernel_size, activation=output_activation, strides=1, padding='valid' )( x )
    return x

def make_pooling( input_layer, channels ):
    x = Conv2DTranspose( channels, kernel_size=(3,3), activation='linear', strides=1, padding='valid' )( input_layer )
    x = make_activation( x )
    x = Conv2D( output_channels, kernel_size=(3,3), activation='linear', strides=2, padding='valid' )( x )
    x = make_activation( x )
    return x

def make_upsampling( input_layer, channels ):
    x = Conv2DTranspose( channels, kernel_size=(4,4), activation='linear', strides=2, padding='valid' )( input_layer )
    x = make_activation( x )
    x = Conv2D( output_channels, kernel_size=(3,3), activation='linear', strides=1, padding='valid' )( x )
    x = make_activation( x )
    return x

# inception
def make_blocks( input_layer, channels, kernel_sizes = ((1, 1), (3, 3), (5, 5), (7, 7)) ):
    layer_blocks = []
    for kernel_size in kernel_sizes:
        layer_blocks.append( make_block( input_layer, channels, kernel_size ) )
    return concatenate( layer_blocks )

def make_bottleneck_block(input_layer, channels=32, scaler=4, kernel_size=(3,3)):
    x = input_layer
    x = make_block( x, channels//scaler, kernel_size=(1,1) )
    x = make_block( x, channels//scaler, kernel_size=kernel_size )
    x = make_block( x, channels, kernel_size=(1,1) )
    return x

def make_residual_block(input_layer, scaler=4):
    *_, output_channels = input_layer.output_shape
    b = make_bottleneck_block( input_layer, output_channels, scaler )
    return add( [input_layer, b] )

def make_resnext_block(input_layer, branches=2, scaler=4):
    *_, output_channels = input_layer.output_shape
    all_layers = []
    for idx in range(branches):
        all_layers.append( make_bottleneck_block( input_layer=input_layer, channels=output_channels, scaler=scaler*branches ) )
    all_layers = add( all_layers )
    return add( [all_layers, input_layer] )

def make_dense_block(input_layer, channels, dims=5, kernel_size=(3,3) ):
    all_layers = input_layer
    for idx in range( dims ):
        x = make_block( input_layer=input_layer, channels=channels, kernel_size=kernel_size )
        all_layers = concatenate( [all_layers, x] )
    return all_layers


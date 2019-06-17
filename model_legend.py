from keras import regularizers
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.models import Model
from keras.layers.advanced_activations import PReLU
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import add
from keras import backend as K
from keras import metrics
from keras.utils import plot_model
from keras.utils import multi_gpu_model
from keras import losses
from keras import optimizers
from keras import metrics
from keras.callbacks import Callback
from keras.models import save_model

def conv2d( conv, input, filters, kernels, bn, strides=2, complexity=8 ):
    x = input
    for idx in range( complexity ):
        if idx == 0:
            x = conv( filters, kernels, strides=strides, padding='same' )( x )
        else:
            x = Conv2D( filters, kernels, strides=1, padding='same' )( x )
        x = LeakyReLU(alpha=0.2)(x)
        if bn:
            x = BatchNormalization(momentum=0.8)(x)
    return x

def downsampling_2d( input, filters, kernels=3, bn=True ):
    return conv2d( Conv2D, input, filters, kernels, bn, strides=2 )

def upsampling_2d( input, filters, kernels=3, bn=True ):
    return conv2d( Conv2DTranspose, input, filters, kernels, bn, strides=2 )

def output_2d( input, filters, kernels, output_channels, output_activation, bn=True ):
    x = Conv2D( filters, kernels, strides=1, padding='same' )( input )
    x = LeakyReLU(alpha=0.2)(x)
    if not bn:
        x = BatchNormalization(momentum=0.8)(x)
    return Conv2D( output_channels, kernel_size=kernels, strides=1, padding='same', activation=output_activation )( x )

def build_model( input_shape=(None, None, 1), output_channels=1, depth=8, output_activation='sigmoid' ) :

    filters = [64, 64, 128, 256, 512, 1024, 1024, 1024, 1024, 1024, 1024]

    init = Input( shape=input_shape )

    e0 = conv2d( Conv2D, init, filters=filters[0], kernels=4, bn=True, strides=1 )
    encoders = [e0, downsampling_2d( init, filters[1], bn=False ),]
    for idx in range( 1, depth ):
        encoders.append( downsampling_2d( encoders[-1], filters[idx+1] ) )

    decoders = [encoders[-1],]
    for idx in range( 1, depth+1 ):
        decoders.append( add( [encoders[-1-idx], upsampling_2d(decoders[-1], filters[depth-idx])] ) )

    outputs = []
    for idx in range( depth ):
        outputs.append( output_2d( decoders[-1-idx], filters=32, kernels=4, output_channels=output_channels, output_activation=output_activation, bn=True ) )

    return Model( inputs=init, outputs=outputs )

if __name__ == '__main__':
    mdcnn = build_model( (512, 512, 1), output_channels=1, depth=3 )
    mdcnn.summary()
    plot_model(mdcnn, 'mdcnn_model.png', show_shapes=True, rankdir='TB')


from keras import regularizers
from keras.layers import Input
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.activations import sigmoid
from keras.activations import tanh
from keras.models import Model
from keras.layers import BatchNormalization
from keras.layers import concatenate
from keras.layers import LeakyReLU
from keras.layers.merge import add
from keras.layers import AveragePooling2D
from keras.utils import plot_model

def build_discriminator( input_shape=(None, None, 1), output_channels=1, regular_factor=0.00001, initializer='he_normal', output_activation=tanh ):

    def make_activation( input_layer ):
        return LeakyReLU(alpha=0.2)(BatchNormalization(momentum=0.8)(input_layer))
        #return LeakyReLU(alpha=0.2)(input_layer)

    def make_block( input_layer, channels, kernel_size=(3,3) ):
        x = input_layer
        x = Conv2DTranspose( channels, kernel_size=kernel_size, activation='linear', strides=1, padding='valid', kernel_regularizer = kr, kernel_initializer = initializer )( x )
        x = make_activation( x )
        x = Conv2D( channels, kernel_size=kernel_size, activation='linear', strides=1, padding='valid', kernel_regularizer = kr, kernel_initializer = initializer )( x )
        x = make_activation( x )
        return x

    def make_output_block( input_layer, twin_channels, kernel_size, output_activation ):
        channels, output_channels_ = twin_channels
        x = input_layer
        x = Conv2DTranspose( channels, kernel_size=kernel_size, activation='linear', strides=1, padding='valid', kernel_regularizer = kr, kernel_initializer = initializer )( x )
        x = make_activation( x )
        x = Conv2D( output_channels_, kernel_size=kernel_size, activation=output_activation, strides=1, padding='valid', kernel_regularizer = kr, kernel_initializer = initializer )( x )
        return x

    def make_pooling( input_layer ):
        return AveragePooling2D(pool_size=(2, 2))(input_layer)

    def make_upsampling( input_layer ):
        return UpSampling2D(size=(2, 2))( input_layer )

    def sum_up( input_layers ):
        return add( input_layers )

    def make_blocks( input_layer, channels, kernel_sizes ):
        sub_channels = int( channels/len(kernel_sizes) )
        assert sub_channels * len(kernel_sizes) == channels, 'sub-channels and channels not match, adjust the channels or the size of sub-kernels'
        layer_blocks = []
        for kernel_size in kernel_sizes:
            layer_blocks.append( make_block( input_layer, sub_channels, kernel_size ) )
        return concatenate( layer_blocks )


    kr = regularizers.l2( regular_factor )
    init = Input( input_shape )

    e_512 = make_blocks( init, 64, ((3, 3), (5, 5),(7, 7), (9, 9) )  )
    e_256 = make_blocks( make_pooling(e_512), 64, ((3, 3), (5, 5), )  )
    e_128 = make_blocks( make_pooling(e_256), 64, ((3, 3), (5, 5), )  )
    e_64  = make_blocks( make_pooling(e_128), 64, ((3, 3), (5, 5), )  )
    e_32  = make_blocks( make_pooling(e_64 ), 128, ((3, 3), (5, 5), )  )
    e_16  = make_blocks( make_pooling(e_32 ), 128, ((3, 3), (5, 5), )  )
    e_8   = make_blocks( make_pooling(e_16 ), 128, ((3, 3), (5, 5), )  )

    o_32  = make_output_block( e_32,  (64, output_channels), (3, 3), output_activation=output_activation )
    o_16  = make_output_block( e_16,  (64, output_channels), (3, 3), output_activation=output_activation )
    o_8   = make_output_block( e_8,   (64, output_channels), (3, 3), output_activation=output_activation )

    model = Model( inputs = init, outputs = [o_32, o_16, o_8] )
    model.summary()

    return model

if __name__ == '__main__':
    #mdcnn = build_discriminator( (512, 512, 2), output_channels = 2 )
    mdcnn = build_discriminator( (720, 1280, 3), output_channels = 1 )
    plot_model(mdcnn, 'mdcnn_discriminator.png', show_shapes=True, rankdir='TB')


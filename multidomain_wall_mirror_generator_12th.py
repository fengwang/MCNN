from keras import regularizers
from keras.layers import Input
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.activations import sigmoid
from keras.layers.convolutional import Cropping2D
from keras.models import Model
from keras.layers import BatchNormalization
from keras.layers import concatenate
from keras.layers import LeakyReLU
from keras.layers.merge import add
from keras.layers import AveragePooling2D
from keras.utils import plot_model
from keras.layers import concatenate

def build_model( input_shape=(None, None, 1), output_channels=1, regular_factor=0.00001, initializer='he_normal', output_activation=sigmoid ):

    def make_activation( input_layer ):
        return LeakyReLU(alpha=0.2)(BatchNormalization(momentum=0.8)(input_layer))

    def make_block( input_layer, channels, kernel_size=(3,3) ):
        x = input_layer
        x = Conv2DTranspose( channels, kernel_size=kernel_size, activation='linear', strides=1, padding='valid', kernel_regularizer = kr, kernel_initializer = initializer )( x )
        x = make_activation( x )
        x = Conv2D( channels, kernel_size=kernel_size, activation='linear', strides=1, padding='valid', kernel_regularizer = kr, kernel_initializer = initializer )( x )
        x = make_activation( x )
        return x

    def make_output_block( input_layer, twin_channels, kernel_size, output_activation ):
        channels, output_channels = twin_channels
        x = input_layer
        x = Conv2DTranspose( channels, kernel_size=kernel_size, activation='linear', strides=1, padding='valid', kernel_regularizer = kr, kernel_initializer = initializer )( x )
        x = make_activation( x )
        x = Conv2D( output_channels, kernel_size=kernel_size, activation=output_activation, strides=1, padding='valid', kernel_regularizer = kr, kernel_initializer = initializer )( x )
        return x

    def make_pooling( input_layer, channels ):
        x = Conv2DTranspose( channels, kernel_size=(3,3), activation='linear', strides=1, padding='valid', kernel_regularizer = kr, kernel_initializer = initializer )( input_layer )
        x = make_activation( x )
        x = Conv2D( output_channels, kernel_size=(3,3), activation='linear', strides=2, padding='valid', kernel_regularizer = kr, kernel_initializer = initializer )( x )
        x = make_activation( x )
        return x
        #return AveragePooling2D(pool_size=(2, 2))(input_layer)

    def make_upsampling( input_layer, channels ):
        x = Conv2DTranspose( channels, kernel_size=(4,4), activation='linear', strides=2, padding='valid', kernel_regularizer = kr, kernel_initializer = initializer )( input_layer )
        x = make_activation( x )
        x = Conv2D( output_channels, kernel_size=(3,3), activation='linear', strides=1, padding='valid', kernel_regularizer = kr, kernel_initializer = initializer )( x )
        x = make_activation( x )
        return x

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

    init = Input( input_shape ) # input of ( 720X1280X3 ), LR flipped

    # init -> pooling
    p_init = make_blocks( make_pooling( init, 32 ), 32, ((1, 1), (3, 3)) )
    # init -> cropping
    #c_init = make_blocks( Cropping2D(cropping=((340, 20), (320, 320)))( init ), 32, (3, 3), (5, 5) )
    c_inits = []
    for r in range( 8 ):
        for c in range( 8 ):
            #c_init = Cropping2D(cropping=((336+r, 16+r), (316+c, 316+c)))( init )
            c_init = Cropping2D(cropping=((316+c, 324-c), (336+r, 24-r)))( init )
            c_inits.append( c_init )

    init_512 = concatenate( c_inits + [p_init,] )

    e_512 = make_blocks( init_512, 64, ((3, 3), (5, 5), (7, 7), (9, 9))  )
    e_256 = make_blocks( make_pooling(e_512, 128), 128, ((3, 3), (5, 5), (7, 7), (9, 9))  )
    e_128 = make_blocks( make_pooling(e_256, 256), 256, ((3, 3), (5, 5), (7, 7), (9, 9))  )
    e_64  = make_blocks( make_pooling(e_128, 256), 256, ((3, 3), (5, 5), (7, 7), (9, 9))  )
    d_64 = e_64
    d_128 = add( [e_128, make_blocks( make_upsampling(d_64, 256 ), 256, ((3, 3), (5, 5), (7, 7), (9, 9))  )] )
    d_256 = add( [e_256, make_blocks( make_upsampling(d_128, 128), 128, ((3, 3), (5, 5), (7, 7), (9, 9))  )] )
    d_512 = add( [e_512, make_blocks( make_upsampling(d_256, 64), 64,  ((3, 3), (5, 5), (7, 7), (9, 9))  )] )
    d_1024 = make_blocks( make_upsampling(d_512, 32), 64,  ((3, 3), (5, 5), (7, 7), (9, 9))  )

    o_1024 = make_output_block( d_1024, (32,  output_channels), (11, 11), output_activation=output_activation )
    o_512 =  make_output_block( d_512,  (32,  output_channels), (9, 9), output_activation=output_activation )
    o_256 =  make_output_block( d_256,  (32,  output_channels), (7, 7), output_activation=output_activation )
    o_128 =  make_output_block( d_128,  (32,  output_channels), (5, 5), output_activation=output_activation )
    o_64 =   make_output_block( d_64,   (32,  output_channels), (3, 3), output_activation=output_activation )

    model = Model( inputs = init, outputs = [o_1024, o_512, o_256, o_128, o_64] )
    model.summary()

    return model

if __name__ == '__main__':
    mdcnn = build_model( (1280, 720, 3), output_channels = 3 )
    #mdcnn = build_model( (640, 360, 3), output_channels = 3 )
    #plot_model(mdcnn, 'new_mdcnn_model.png', show_shapes=True, rankdir='TB')


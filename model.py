from keras import regularizers
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization
from keras.models import Sequential
from keras.models import Model
from keras.layers.advanced_activations import PReLU
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

def bn_patch( input_layer, with_bn ):
    if with_bn:
        return BatchNormalization(momentum=0.8)(input_layer)
    return input_layer

def build_model( img_rows=None, img_cols=None, img_channels=1, output_channels=1, regular_factor=0.00001, initializer='he_normal', output_activation='sigmoid', with_bn=False ):
    original_img_size = ( img_rows, img_cols, img_channels )
    kr = regularizers.l2( regular_factor )

    init = Input( shape=(img_rows, img_cols, img_channels) )
    init_ = Conv2D( 32, kernel_size=(31, 31), activation='relu', strides = 1, padding='same', kernel_regularizer=kr, kernel_initializer=initializer )( init )
    init_ = bn_patch( init_, with_bn )
    l1 = Conv2D( 64, kernel_size=(17, 17), activation='relu', strides = 2, padding='same', kernel_regularizer=kr, kernel_initializer=initializer )( init_ ) # 256
    l1 = bn_patch( l1, with_bn )
    l2 = Conv2D(128, kernel_size=(11, 11), activation= 'relu', strides=2, padding='same', kernel_regularizer=kr, kernel_initializer=initializer )( l1 ) # 128
    l2 = bn_patch( l2, with_bn )
    l3 = Conv2D(192, kernel_size=(5, 5), activation='relu', strides=2, padding='same', kernel_regularizer=kr, kernel_initializer=initializer )( l2 ) # 64
    l3 = bn_patch( l3, with_bn )
    l4 = Conv2D(256, kernel_size=(3, 3), activation='relu', strides=2, padding='same', kernel_regularizer=kr, kernel_initializer=initializer )( l3 ) # 32
    l4 = bn_patch( l4, with_bn )
    l5 = Conv2D(384, kernel_size=(3, 3), activation='relu', strides=2, padding='same', kernel_regularizer=kr, kernel_initializer=initializer )( l4 ) # 16
    l5 = bn_patch( l5, with_bn )
    l6 = Conv2D(512, kernel_size=(3, 3), activation='relu', strides=2, padding='same', kernel_regularizer=kr, kernel_initializer=initializer )( l5 ) # 8
    l6 = bn_patch( l6, with_bn )

    lx1 = Conv2D(1024, kernel_size=(3, 3), activation='relu', strides=2, padding='same', kernel_regularizer=kr, kernel_initializer=initializer )( l6 ) # 4
    lx1 = bn_patch( lx1, with_bn )
    lx2 = Conv2DTranspose( 1024, kernel_size=(3,3), activation='relu', strides=1, padding='valid' )( lx1 )
    lx2 = bn_patch( lx2, with_bn )
    lx3 = Conv2D( 1024, kernel_size=(3,3), activation='relu', strides=1, padding='valid' )( lx2 ) # 4
    lx3 = bn_patch( lx3, with_bn )

    #lx1 = Conv2D(768, kernel_size=(3, 3), activation='relu', strides=(1,1), padding='valid', kernel_regularizer=kr, kernel_initializer=initializer )( l6 ) # 6
    #lx1 = Conv2D(1024, kernel_size=(3, 3), activation='relu', strides=(1,1), padding='valid', kernel_regularizer=kr, kernel_initializer=initializer )( lx1 ) # 4
    #lx2 = Conv2D(2048, kernel_size=(3, 3), activation='relu', strides=(1,1), padding='valid', kernel_regularizer=kr, kernel_initializer=initializer )( lx1 ) # 2
    #lx3 = Conv2DTranspose(1024, kernel_size=(3, 3), strides=(1,1), padding='valid', kernel_regularizer=kr, kernel_initializer=initializer, activation='relu' )( lx2 ) # 4

    lx4 = add( [lx3, lx1] )

    #lx5 = Conv2DTranspose(768, kernel_size=(3, 3), strides=(1,1), padding='valid', kernel_regularizer=kr, kernel_initializer=initializer, activation='relu' )( lx4 ) # 6
    #l7 = Conv2DTranspose(512, kernel_size=(3, 3), strides=(1,1), padding='valid', kernel_regularizer=kr, kernel_initializer=initializer, activation='relu' )( lx5 ) # 8
    l7 = Conv2DTranspose(512, kernel_size=(3, 3), strides=(2,2), padding='same', kernel_regularizer=kr, kernel_initializer=initializer, activation='relu' )( lx4 ) # 8
    l7 = bn_patch( l7, with_bn )
    l8 = l7
    l9 = add( [l8, l6] ) #

    l10 = Conv2DTranspose(384, kernel_size=(3, 3), strides=(2,2), padding='same', kernel_regularizer=kr, kernel_initializer=initializer, activation='relu' )( l9 ) # 16
    l10 = bn_patch( l10, with_bn )
    l11 = l10
    l12 = add( [l11, l5] ) #

    l13 = Conv2DTranspose(256, kernel_size=(3, 3),  activation='relu', strides=(2,2), padding='same', kernel_regularizer=kr, kernel_initializer=initializer )( l12 ) # 32
    l13 = bn_patch( l13, with_bn )
    l14 = l13
    l15 = add( [l14, l4] ) #

    l16 = Conv2DTranspose(192, kernel_size=(3, 3), strides=(2,2), activation='relu', padding='same', kernel_regularizer=kr, kernel_initializer=initializer )( l15 ) # 64
    l16 = bn_patch( l16, with_bn )
    l17 = l16
    l18 = add( [l17, l3] ) #

    l19 = Conv2DTranspose(128, kernel_size=(3, 3), activation='relu', strides=(2,2), padding='same', kernel_regularizer=kr, kernel_initializer=initializer )( l18 ) # 128
    l19 = bn_patch( l19, with_bn )
    l20 = l19
    l21 = add( [l20, l2] ) #

    l22 = Conv2DTranspose(64, kernel_size=(3, 3), activation='relu', strides=(2,2), padding='same', kernel_regularizer=kr, kernel_initializer=initializer )( l21 ) # 256
    l22 = bn_patch( l22, with_bn )
    l23 = l22
    l24 = add( [l23, l1] )

    l25 = Conv2DTranspose(32, kernel_size=(3, 3), activation='relu', strides=(2,2), padding='same', kernel_regularizer=kr, kernel_initializer=initializer )( l24 ) # 512
    l25 = bn_patch( l25, with_bn )
    l26 = l25
    l27 = add( [l26, init_] )

    llast = Conv2D(16 , kernel_size=(5, 5), activation='relu', strides=(1,1), padding='same', kernel_regularizer=kr, kernel_initializer=initializer )( l27 )
    llast = bn_patch( llast, with_bn )
    last512 = Conv2D(output_channels , kernel_size=(31, 31), activation=output_activation, strides=(1,1), padding='same', kernel_regularizer=kr, kernel_initializer=initializer, name="gen_512" )( llast )

    last4 = Conv2D(16 , kernel_size=(3, 3), activation='relu', strides=(1,1), padding='same', kernel_regularizer=kr, kernel_initializer=initializer )( lx4 )
    last4 = bn_patch( last4, with_bn )
    last4 = Conv2D(output_channels , kernel_size=(3, 3), activation=output_activation, strides=(1,1), padding='same', kernel_regularizer=kr, kernel_initializer=initializer, name="gen_4")( last4 )

    last8 = Conv2D(16 , kernel_size=(3, 3), activation='relu', strides=(1,1), padding='same', kernel_regularizer=kr, kernel_initializer=initializer )( l9 )
    last8 = bn_patch( last8, with_bn )
    last8 = Conv2D(output_channels , kernel_size=(5, 5), activation=output_activation, strides=(1,1), padding='same', kernel_regularizer=kr, kernel_initializer=initializer, name="gen_8")( last8 )

    last16 = Conv2D(16 , kernel_size=(3, 3), activation='relu', strides=(1,1), padding='same', kernel_regularizer=kr, kernel_initializer=initializer )( l12 )
    last16 = bn_patch( last16, with_bn )
    last16 = Conv2D(output_channels , kernel_size=(5, 5), activation=output_activation, strides=(1,1), padding='same', kernel_regularizer=kr, kernel_initializer=initializer, name="gen_16")( last16 )

    last32 = Conv2D(16 , kernel_size=(3, 3), activation='relu', strides=(1,1), padding='same', kernel_regularizer=kr, kernel_initializer=initializer )( l15 )
    last32 = bn_patch( last32, with_bn )
    last32 = Conv2D(output_channels , kernel_size=(7, 7), activation=output_activation, strides=(1,1), padding='same', kernel_regularizer=kr, kernel_initializer=initializer, name="gen_32" )( last32 )

    last64 = Conv2D(16 , kernel_size=(3, 3), activation='relu', strides=(1,1), padding='same', kernel_regularizer=kr, kernel_initializer=initializer )( l18 )
    last64 = bn_patch( last64, with_bn )
    last64 = Conv2D(output_channels , kernel_size=(9, 9), activation=output_activation, strides=(1,1), padding='same', kernel_regularizer=kr, kernel_initializer=initializer, name="gen_64" )( last64 )

    last128 = Conv2D(16 , kernel_size=(3, 3), activation='relu', strides=(1,1), padding='same', kernel_regularizer=kr, kernel_initializer=initializer )( l21 )
    last128 = bn_patch( last128, with_bn )
    last128 = Conv2D(output_channels , kernel_size=(11, 11), activation=output_activation, strides=(1,1), padding='same', kernel_regularizer=kr, kernel_initializer=initializer, name="gen_128" )( last128 )

    last256 = Conv2D(16 , kernel_size=(3, 3), activation='relu', strides=(1,1), padding='same', kernel_regularizer=kr, kernel_initializer=initializer )( l24 )
    last256 = bn_patch( last256, with_bn )
    last256 = Conv2D(output_channels , kernel_size=(17, 17), activation=output_activation, strides=(1,1), padding='same', kernel_regularizer=kr, kernel_initializer=initializer, name="gen_256" )( last256 )

    return Model( inputs = init, outputs = [last512, last256, last128, last64, last32, last16, last8, last4] )

if __name__ == '__main__':
    #unet, mdcnn = build_model( 1920, 1080, img_channels = 2 )
    mdcnn = build_model( 768, 768, img_channels=4, output_channels=2 )
    #plot_model(unet, 'unet_model.png', show_shapes=True, rankdir='TB')
    print( mdcnn.summary() )
    plot_model(mdcnn, 'mdcnn_model.png', show_shapes=True, rankdir='TB')


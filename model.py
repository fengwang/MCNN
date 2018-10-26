from keras import regularizers
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
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

def build_model( img_rows=None, img_cols=None, img_channel=1, regular_factor=0.00001 ):

    original_img_size = ( img_rows, img_cols, img_channel )

    initializer = "he_normal"

    kr = regularizers.l2( regular_factor )

    init = Input( shape=(img_rows, img_cols, img_channel ) )
    l1 = Conv2D( 64, kernel_size=(7, 7), activation = "relu", strides = 2, padding="same", kernel_regularizer = kr, init = initializer )( init ) # 256
    l2 = Conv2D(128, kernel_size=(5, 5), activation= "relu", strides=2, padding='same', kernel_regularizer = kr )( l1 ) # 128
    l3 = Conv2D(192, kernel_size=(3, 3), activation="relu", strides=2, padding='same', kernel_regularizer = kr )( l2 ) # 64
    l4 = Conv2D(256, kernel_size=(3, 3), activation="relu", strides=2, padding='same', kernel_regularizer = kr )( l3 ) # 32
    l5 = Conv2D(384, kernel_size=(3, 3), activation="relu", strides=2, padding='same', kernel_regularizer = kr )( l4 ) # 16
    l6 = Conv2D(512, kernel_size=(3, 3), activation="relu", strides=2, padding='same', kernel_regularizer = kr )( l5 ) # 8

    lx1 = Conv2D(768, kernel_size=(3, 3), activation="relu", strides=(1,1), padding='valid', kernel_regularizer = kr )( l6 ) # 6
    lx1 = Conv2D(1024, kernel_size=(3, 3), activation="relu", strides=(1,1), padding='valid', kernel_regularizer = kr )( lx1 ) # 4
    lx2 = Conv2D(2048, kernel_size=(3, 3), activation="relu", strides=(1,1), padding='valid', kernel_regularizer = kr )( lx1 ) # 2
    lx3 = Conv2DTranspose(1024, kernel_size=(3, 3), strides=(1,1), padding='valid', kernel_regularizer = kr, activation="relu" )( lx2 ) # 4

    lx4 = add( [lx3, lx1] )

    lx5 = Conv2DTranspose(768, kernel_size=(3, 3), strides=(1,1), padding='valid', kernel_regularizer = kr, activation="relu" )( lx4 ) # 6
    l7 = Conv2DTranspose(512, kernel_size=(3, 3), strides=(1,1), padding='valid', kernel_regularizer = kr, activation="relu" )( lx5 ) # 8
    l8 = l7
    l9 = add( [l8, l6] ) #

    l10 = Conv2DTranspose(384, kernel_size=(3, 3), strides=(2,2), padding='same', kernel_regularizer = kr, activation = "relu" )( l9 ) # 16
    l11 = l10
    l12 = add( [l11, l5] ) #

    l13 = Conv2DTranspose(256, kernel_size=(3, 3),  activation="relu", strides=(2,2), padding='same', kernel_regularizer = kr )( l12 ) # 32
    l14 = l13
    l15 = add( [l14, l4] ) #

    l16 = Conv2DTranspose(192, kernel_size=(3, 3), strides=(2,2), activation = "relu", padding='same', kernel_regularizer = kr )( l15 ) # 64
    l17 = l16
    l18 = add( [l17, l3] ) #

    l19 = Conv2DTranspose(128, kernel_size=(3, 3), activation = "relu", strides=(2,2), padding='same', kernel_regularizer = kr )( l18 ) # 128
    l20 = l19
    l21 = add( [l20, l2] ) #

    l22 = Conv2DTranspose(64, kernel_size=(3, 3), activation = "relu", strides=(2,2), padding='same', kernel_regularizer = kr )( l21 ) # 256
    l23 = l22
    l24 = add( [l23, l1] )

    l25 = Conv2DTranspose(32, kernel_size=(3, 3), activation = "relu", strides=(2,2), padding='same', kernel_regularizer = kr )( l24 ) # 512
    l26 = l25
    l27 = add( [l26, init] )

    llast = Conv2D(16 , kernel_size=(3, 3), activation='relu', strides=(1,1), padding='same', kernel_regularizer = kr )( l27 )
    llast = Conv2D(8 , kernel_size=(3, 3), activation='relu', strides=(1,1), padding='same', kernel_regularizer = kr )( llast )
    last512 = Conv2D(1 , kernel_size=(31, 31), activation='sigmoid', strides=(1,1), padding='same', kernel_regularizer = kr, name="gen_512" )( llast )

    last4 = Conv2D(8 , kernel_size=(3, 3), activation='relu', strides=(1,1), padding='same', kernel_regularizer = kr )( lx4 )
    last4 = Conv2D(1 , kernel_size=(3, 3), activation='sigmoid', strides=(1,1), padding='same', kernel_regularizer = kr, name="gen_4")( last4 )

    last8 = Conv2D(8 , kernel_size=(3, 3), activation='relu', strides=(1,1), padding='same', kernel_regularizer = kr )( l9 )
    last8 = Conv2D(1 , kernel_size=(5, 5), activation='sigmoid', strides=(1,1), padding='same', kernel_regularizer = kr, name="gen_8")( last8 )

    last16 = Conv2D(8 , kernel_size=(3, 3), activation='relu', strides=(1,1), padding='same', kernel_regularizer = kr )( l12 )
    last16 = Conv2D(1 , kernel_size=(5, 5), activation='sigmoid', strides=(1,1), padding='same', kernel_regularizer = kr, name="gen_16")( last16 )

    last32 = Conv2D(8 , kernel_size=(3, 3), activation='relu', strides=(1,1), padding='same', kernel_regularizer = kr )( l15 )
    last32 = Conv2D(1 , kernel_size=(7, 7), activation='sigmoid', strides=(1,1), padding='same', kernel_regularizer = kr, name="gen_32" )( last32 )

    last64 = Conv2D(8 , kernel_size=(3, 3), activation='relu', strides=(1,1), padding='same', kernel_regularizer = kr )( l18 )
    last64 = Conv2D(1 , kernel_size=(9, 9), activation='sigmoid', strides=(1,1), padding='same', kernel_regularizer = kr, name="gen_64" )( last64 )

    last128 = Conv2D(8 , kernel_size=(3, 3), activation='relu', strides=(1,1), padding='same', kernel_regularizer = kr )( l21 )
    last128 = Conv2D(1 , kernel_size=(11, 11), activation='sigmoid', strides=(1,1), padding='same', kernel_regularizer = kr, name="gen_128" )( last128 )

    last256 = Conv2D(8 , kernel_size=(3, 3), activation='relu', strides=(1,1), padding='same', kernel_regularizer = kr )( l24 )
    last256 = Conv2D(1 , kernel_size=(17, 17), activation='sigmoid', strides=(1,1), padding='same', kernel_regularizer = kr, name="gen_256" )( last256 )

    pre_last256 = Conv2D(8 , kernel_size=(3, 3), activation='relu', strides=(1,1), padding='same', kernel_regularizer = kr )( l24 )
    pre_last256 = Conv2D(1 , kernel_size=(17, 17), activation='sigmoid', strides=(1,1), padding='same', kernel_regularizer = kr, name="pre_256" )( pre_last256 )

    normal_model = Model( inputs = init, outputs = [last512, last256, last128, last64, last32, last16, last8, last4] )
    generator_model = Model( inputs = init, outputs = last512 )

    #final_model = multi_gpu_model( normal_model, gpus=2 )
    final_model = normal_model

    return (generator_model, final_model)


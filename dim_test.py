from keras import regularizers
from keras.layers import Input
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.activations import sigmoid
from keras.models import Model
from keras.layers import BatchNormalization
from keras.layers import concatenate
from keras.layers import LeakyReLU
from keras.layers.merge import add
from keras.layers import AveragePooling2D
from keras.utils import plot_model

def make_activation( input_layer ):
    return LeakyReLU(alpha=0.2)(BatchNormalization(momentum=0.8)(input_layer))

def build_model( input_shape=(512, 512, 1), channels=16,  output_channels=1, regular_factor=0.00001, initializer='he_normal', output_activation=sigmoid ):
    input = Input( input_shape )
    x = Conv2DTranspose( channels, kernel_size=(3,3), activation='linear', strides=1, padding='valid' )( input )
    x = make_activation( x )
    x = Conv2D( output_channels, kernel_size=(3,3), activation='linear', strides=2, padding='valid' )( x )
    x = make_activation( x )

    m = Model( input, x )
    m.summary()
    return m

if __name__ == '__main__':
    m = build_model()
    m = build_model(input_shape=(640, 360, 3))


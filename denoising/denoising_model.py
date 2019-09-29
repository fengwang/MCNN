from keras.layers import Input, Concatenate
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.utils import plot_model

def conv2d(layer_input, filters, f_size=4, bn=False):
    d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
    d = LeakyReLU(alpha=0.2)(d)
    if bn:
        d = BatchNormalization(momentum=0.8)(d)
    return d

def deconv2d(layer_input, skip_input, filters, f_size=4):
    u = UpSampling2D(size=2)(layer_input)
    u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
    u = BatchNormalization(momentum=0.8)(u)
    u = Concatenate()([u, skip_input])
    return u

def build_denoising_model(input_shape=(None,None,4)):

    d0 = Input(shape=input_shape)
    d1 = conv2d(d0, 64, bn=False)
    d2 = conv2d(d1, 128)
    d3 = conv2d(d2, 256)
    d4 = conv2d(d3, 512)
    d5 = conv2d(d4, 512)
    d6 = conv2d(d5, 512)
    d7 = conv2d(d6, 512)

    u1 = deconv2d(d7, d6, 512)
    u2 = deconv2d(u1, d5, 512)
    u3 = deconv2d(u2, d4, 512)
    u4 = deconv2d(u3, d3, 256)
    u5 = deconv2d(u4, d2, 128)
    u6 = deconv2d(u5, d1, 64)
    u7 = UpSampling2D(size=2)(u6)

    last_512 = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='sigmoid')(u7)
    last_256 = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='sigmoid')(u6)
    last_128 = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='sigmoid')(u5)
    last_64  = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='sigmoid')(u4)
    last_32  = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='sigmoid')(u3)
    last_16  = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='sigmoid')(u2)
    last_8   = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='sigmoid')(u1)

    #u_model     = Model(d0, last_512)
    mcnn_model  = Model(d0, outputs=[last_512, last_256, last_128, last_64, last_32, last_16, last_8])
    unet_model = Model( d0,  last_512 )
    return unet_model, mcnn_model

if __name__ == '__main__':
    u, m = build_denoising_model(input_shape=(512,512,4))
    #plot_model(u, 'denoising_u_model.png', show_shapes=True, rankdir='TB')
    plot_model(m, 'denoising_model.png', show_shapes=True, rankdir='TB')
    m.summary()


from denoising_model import build_denoising_model
from discriminator import build_discriminator
from data_loader import DataLoader
from train import mcnn_train
from lpf_model import merge_model
from keras.optimizers import Adam

if __name__ == '__main__':
    mcnn_model = build_denoising_model()
    discriminator_model = build_discriminator()
    loader = DataLoader( 2048 )

    config = {
        'epochs': 4096 * 2048 // 4, # batches
        'batch_size': 4,
        'optimizer': Adam(lr=0.0001, decay=0.0001),
        'sampling_folder': './sampling',
        'sampling_interval': 32,
        'validation_interval': 4
    }

    mcnn_train(mcnn_model, discriminator_model, loader, config )

    mcnn_model.save( './mcnn_model.h5' )
    merge_model( './mcnn_model.h5', './denoising_model.h5' )


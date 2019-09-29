from denoising_model import build_denoising_model
from discriminator import build_discriminator
from data_loader import DataLoader
from train import mcnn_train
from lpf_model import merge_model
from keras.optimizers import Adam

if __name__ == '__main__':
    unet_model, mcnn_model = build_denoising_model()
    discriminator_model = build_discriminator()
    loader = DataLoader( 2048 )

    config = {
        'epochs': 4096 * 2048 // 4, # batches
        'batch_size': 4,
        'optimizer': Adam(lr=0.0001, decay=0.0001),
        'sampling_folder': './sampling',
        'sampling_interval': 512,
        'validation_interval': 386
    }

    mcnn_train(mcnn_model, discriminator_model, loader, config )

    # deserting low frequency branches
    unet_model.save( './denoising.model' )
    merge_model( './denoising.model', './denoising_merged.model' )


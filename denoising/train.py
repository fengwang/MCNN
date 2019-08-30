import numpy as np
from keras.layers import Input
from keras.models import Model
from keras.utils import multi_gpu_model
from skimage.measure import block_reduce
import imageio

def generate_low_frequency_components( high_frequency_input, levels ):
    ans = [high_frequency_input,]
    for idx in range( levels-1 ):
        ans.append( block_reduce(ans[-1], block_size=(1, 2, 2, 1), func=np.mean) )
    return ans

def dump_data( data, identity ):
    data = np.squeeze( data )
    if len(data.shape) == 2:
        imageio.imsave( f'{identity}.png', data )
        return
    *_, n = data.shape
    for idx in range( n ):
        dump_data( data[:,:,idx], f'{identity}_{idx}' )

# training config:
#       batch_size
#       optimizer
#       sampling_folder
#       sampling_interval
#       validation_interval
def mcnn_train( mcnn_model, discriminator_model, data_loader, training_config ):
    optimizer = training_config['optimizer']
    batch_size = training_config['batch_size']
    epochs = training_config['epochs']

    mcnn_output_numbers = len( mcnn_model.outputs )
    model_to_train = mcnn_model

    discriminator_output_numbers = 0

    # case of gan
    if discriminator_model is not None:
        discriminator_output_numbers = len( discriminator_model.outputs )
        sample_input, sample_output = data_loader.load_batch( 1 ) # sample_output
        row, col, *_ = np.squeeze( sample_output ).shape

        # generate outputs for GAN
        fake = np.zeros( (batch_size, row//16, col//16, 1) )
        fakes = generate_low_frequency_components( fake, discriminator_output_numbers )
        real = np.ones( (batch_size, row//16, col//16, 1) )
        reals = generate_low_frequency_components( real, discriminator_output_numbers )

        # define a combined model
        mcnn_input_channels = (mcnn_model._feed_input_shapes)[0][-1]
        mcnn_input_layer = Input( ( None, None, mcnn_input_channels ) )
        mcnn_output_layers = mcnn_model( mcnn_input_layer )
        discriminator_input_layer, *_ = mcnn_output_layers
        #discriminator = multi_gpu_model( discriminator_model, gpus=2 )
        discriminator = discriminator_model
        discriminator.compile( loss='mse', optimizer=optimizer, metrics=['accuracy'] )
        discriminator.Trainable = False
        discriminator_output_layers = discriminator_model( discriminator_input_layer )
        combined_output_layers = mcnn_output_layers + discriminator_output_layers
        combined_model = Model( mcnn_input_layer, combined_output_layers )

        model_to_train = combined_model

    #final_model = multi_gpu_model( model_to_train, gpus=2 )
    final_model = model_to_train
    final_model_loss = ['mae',]*mcnn_output_numbers + ['mse',]*discriminator_output_numbers
    final_model.summary()
    final_model.compile( loss=final_model_loss, optimizer=optimizer )


    for idx in range( epochs ):
        input_data, output_data = data_loader.load_batch( batch_size )
        multi_scale_output_data = generate_low_frequency_components( output_data, mcnn_output_numbers )

        if discriminator_model is not None:
            fake_prediction, *_ = mcnn_model.predict( input_data )
            discriminator_loss_fake = discriminator.train_on_batch( fake_prediction, fakes )
            discriminator_loss_real = discriminator.train_on_batch( output_data, reals )
            final_model_loss = final_model.train_on_batch( input_data, multi_scale_output_data+reals )
            loss_so_far = discriminator_loss_fake + discriminator_loss_real + final_model_loss
        else:
            loss_so_far = final_model.train_on_batch( input_data, multi_scale_output_data )

        print( f'epoch {idx}/{epochs}: {loss_so_far}' )

        if 'sampling_interval' in training_config:
            sampling_interval = training_config['sampling_interval']
            if ( idx != 0 ) and ( idx % sampling_interval == 0 ):
                input_data_sampling, output_data_sampling = data_loader.load_batch( 1 )
                prediction, *_ = mcnn_model.predict( input_data_sampling )
                sampling_folder = training_config['sampling_folder']
                dump_data( input_data_sampling, f'{sampling_folder}/input_{idx}' )
                dump_data( output_data_sampling, f'{sampling_folder}/output_{idx}' )
                dump_data( prediction, f'{sampling_folder}/prediction_{idx}' )
                mcnn_model.save( f'{sampling_folder}/model_{idx}.h5' )

        validation_interval = 4
        if 'validation_interval' in training_config:
            validation_interval = training_config['validation_interval']
        if ( idx != 0 ) and ( idx%validation_interval == 0 ):
            input_data_validation, output_data_validation = data_loader.load_validation_batch( batch_size )
            #print( f'validation data loaded with shape {input_data_validation.shape} and {output_data_validation.shape}' )
            prediction, *_ = mcnn_model.predict(input_data_validation)
            validation_diff = np.abs( np.squeeze(output_data_validation) - np.squeeze(prediction) )
            print( f'Validation at {idx}/{epochs} MAE loss {np.mean(validation_diff)}' )


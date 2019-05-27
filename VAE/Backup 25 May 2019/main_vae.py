# Basic VAE MNIST - main file
# Adapted from https://keras.io/examples/variational_autoencoder_deconv/
# Potential improvements:
#  - Extend to conditional VAE
#  - Low priority: Tackle checkerboard artifacts due to transposed convolution
#    https://stackoverflow.com/questions/45559846/how-to-remove-deconvolution-noise-in-style-transfer-neural-network
import math
import numpy as np
import vae_utils

from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.models import load_model
from keras.utils import plot_model

# Hyperparameters
mode = ['train', 'inspect', 'train_inspect'][2]
save_path_encoder = 'vae_encoder.h5'
save_path_decoder = 'vae_decoder.h5'
hyperpars = {
    'batch_size': 32,
    'num_epochs': 10,
    'initial_lr': 1e-3,
    'reduce_lr_patience': 3,
    'es_patience': 5,
    
    'filters_kernels_strides': [(32, 3, 2), (16, 3, 2)],
#    'filters_kernels_strides': [(32, 3, 1), (32, 3, 1), (32, 3, 1), (32, 3, 2)],
    'latent_mlp_layers': [64, 16],
    'latent_dim': 16,
    
    'kl_beta': 5, # Beta-VAE https://openreview.net/pdf?id=Sy2fzU9gl
    }

if not 'images' in locals():
  images = np.expand_dims(np.load('../Data/training_images.npy'), -1)/255.
  x_test = np.expand_dims(np.load('../Data/test_images.npy'), -1)/255.
  y_test = np.load('../Data/test_labels.npy')
  
if 'train' in mode:
  K.clear_session()
  (model, encoder, decoder, custom_metrics) = vae_utils.mnist_vae(hyperpars)
  adam = Adam(lr=hyperpars['initial_lr'])
  model.compile(optimizer=adam)
  vae_utils.add_custom_metrics(model, custom_metrics)
  plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)
  plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)
  plot_model(model, to_file='vae_cnn.png', show_shapes=True)
  (monitor, monitor_mode) = ('loss', 'min')
  earlystopper = EarlyStopping(
      monitor=monitor, mode=monitor_mode,
      patience=hyperpars['es_patience'], verbose=1)
  encoder_checkpointer = vae_utils.CustomCheckpointer(
        save_path_encoder, encoder, monitor=monitor, mode=monitor_mode,
        save_best_only=True, verbose=0, verbose_description='encoder')
  decoder_checkpointer = vae_utils.CustomCheckpointer(
        save_path_decoder, decoder, monitor=monitor, mode=monitor_mode,
        save_best_only=True, verbose=0, verbose_description='decoder')
  reduce_lr = ReduceLROnPlateau(factor=1/math.sqrt(10), verbose=1,
                                patience=hyperpars['reduce_lr_patience'],
                                min_lr=hyperpars['initial_lr']/100,
                                monitor=monitor,
                                mode=monitor_mode)
  loss_plotter = vae_utils.PlotLosses()
  callbacks = [earlystopper, encoder_checkpointer, decoder_checkpointer,
               reduce_lr, loss_plotter]
  model.fit(images, batch_size=hyperpars['batch_size'],
            epochs=hyperpars['num_epochs'], callbacks=callbacks)
  
if 'inspect' in mode:
  K.clear_session()
  encoder = load_model(save_path_encoder)
  decoder = load_model(save_path_decoder)
  vae_utils.plot_results((encoder, decoder), (x_test, y_test),
                         latent_dim=hyperpars['latent_dim'])
# Basic GAN MNIST - main file
# Inspired by https://medium.com/datadriveninvestor/generative-adversarial-network-gan-using-keras-ce1c05cfdfd3
# Loosely inspired by https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0
# Potential next steps:
#   - Conditional digit MLP encoder instead of simple one-hot encoding
#   - Non symmetric Generator-Discriminator structure
#   - Use upsampling (UpSampling2D) instead of fractionally-strided convolution
import numpy as np
import utils_gan

from keras import backend as K
from keras.optimizers import Adam
from keras.models import load_model
from keras.utils import plot_model
from keras.utils import to_categorical

# Execution and model (hyper)parameters
mode = ['train', 'inspect', 'train_inspect'][2]
save_path_generator = 'gan_generator.h5'
save_path_discriminator = 'gan_discriminator.h5'
hyperpars = {
    'batch_size': 32,
    'num_epochs': 100,
    'initial_lr_discriminator': 4e-4, # Makes sense since it sees double data
    'initial_lr_generator': 8e-4,
    'true_data_target_prob': 0.9,
    
    'filters_kernels_strides': [(32, 3, 2), (16, 3, 2)],
#    'filters_kernels_strides': [(32, 3, 2), (16, 3, 1)],
#    'filters_kernels_strides': [(32, 3, 1), (32, 3, 1), (32, 3, 1), (32, 3, 2), (16, 3, 2)],
    'latent_mlp_layers': [64, 16],
    'latent_dim': 16,
    
    'conditional_gan': True,
    }

if not 'images' in locals():
  images = np.expand_dims(np.load('../Data/training_images.npy'), -1)/255.
  labels = np.load('../Data/training_labels.npy').astype(np.int32)
  labels_onehot = to_categorical(labels, 10).astype(np.float32)
  x_test = np.expand_dims(np.load('../Data/test_images.npy'), -1)/255.
  y_test = np.load('../Data/test_labels.npy').astype(np.int32)
  y_test_onehot = to_categorical(y_test, 10).astype(np.float32)
  
if 'train' in mode:
  K.clear_session()
  (discriminator, generator, adversarial_prob_model) = utils_gan.mnist_gan(
      hyperpars)
  adam_discriminator = Adam(lr=hyperpars['initial_lr_discriminator'])
  discriminator.compile(optimizer=adam_discriminator,
                        loss='binary_crossentropy')
  adam_generator = Adam(lr=hyperpars['initial_lr_generator'])
#  generator.compile(optimizer=adam_generator, loss='binary_crossentropy')
  discriminator.trainable = False # CRUCIAL!
  adversarial_prob_model.compile(optimizer=adam_generator,
                                 loss='binary_crossentropy')
  plot_model(generator, to_file='gan_cnn_generator.png', show_shapes=True)
  plot_model(discriminator, to_file='gan_cnn_discriminator.png',
             show_shapes=True)
  plot_model(adversarial_prob_model, to_file='gan_cnn_adv.png',
             show_shapes=True)
  train_gen = utils_gan.conditional_gan_generator(
      images, labels_onehot, hyperpars)
  utils_gan.my_gan_fit(train_gen, discriminator, generator,
                       adversarial_prob_model, hyperpars)
  generator.save(save_path_generator)
  discriminator.save(save_path_discriminator)
  
if 'inspect' in mode:
  K.clear_session()
  generator = load_model(save_path_generator)
  train_gen = utils_gan.conditional_gan_generator(
      images, labels_onehot, hyperpars)
  utils_gan.plot_generated_images('inspect', train_gen, generator, hyperpars)
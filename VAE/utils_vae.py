# Basic VAE MNIST - utilities
# Adapted from https://keras.io/examples/variational_autoencoder_deconv/
import keras
from keras import backend as K
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Reshape
from keras.losses import mse
from keras.models import Model
from keras.utils import to_categorical

import matplotlib.pyplot as plt
import numpy as np
import os

# Input data generator
def conditional_vae_generator(images, labels, batch_size):
  steps_per_epoch=images.shape[0] // batch_size
  while True:
    shuffled_ids = np.random.permutation(images.shape[0])
    for i in range(steps_per_epoch):
      batch_images = images[shuffled_ids[i*batch_size:((i+1)*batch_size)]]
      batch_labels = labels[shuffled_ids[i*batch_size:((i+1)*batch_size)]]
      
      # Two inputs, no output
      yield [batch_images, batch_labels], [] 


# Reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + z_std*eps
def sample_gaussian(args):
  z_mean, z_std = args
  batch = K.shape(z_mean)[0]
  dim = K.int_shape(z_mean)[1]
  epsilon = K.random_normal(shape=(batch, dim))
  return z_mean + z_std * epsilon


# VAE loss - reconstruction + beta*kl_loss
def vae_loss(z_mean, z_std, inputs, outputs, image_size, hyperpars):
  rec_loss = mse(K.flatten(inputs), K.flatten(outputs))
  rec_loss = K.mean(rec_loss*image_size*image_size)
  rec_loss = Lambda(lambda x: x, name='reconstruction_loss')(rec_loss)
  
  kl_loss = 1 + K.log(K.square(z_std)) - K.square(z_mean) - K.square(z_std)
#  kl_loss = 1/2*(K.square(z_std) + K.square(z_mean) - 2*K.log(z_std) - 1)
  kl_loss = K.mean(K.sum(kl_loss, axis=-1)*-1/hyperpars['latent_dim'])
  kl_loss *= hyperpars['kl_beta']
  kl_loss = Lambda(lambda x: x, name='kl_loss')(kl_loss)
  
  metrics = {'reconstruction_loss': rec_loss, 'kl_loss': kl_loss}
  
  return [rec_loss, kl_loss], metrics


# MNIST autoencoder model
def mnist_vae(hyperpars):
  image_size = 28
  # 1) Encoder
  image_inputs = Input((image_size, image_size, 1), name='image_inputs')
  label_inputs = Input((10,), dtype='float32', name='label_inputs')
  x = image_inputs
  for (filters, kernel, strides) in hyperpars['filters_kernels_strides']:
    x = Conv2D(filters=filters, kernel_size=kernel, strides=strides,
               padding='same')(x)
  shape = K.int_shape(x) # shape info needed to build decoder model
  print('Shape after convolution: {}'.format(shape))
  x = Flatten()(x)
  if hyperpars['conditional_vae']:
    x = Lambda(lambda x: K.concatenate(x, axis=-1))([x, label_inputs])
  for layer_size in hyperpars['latent_mlp_layers']:
    x = Dense(layer_size, activation='relu')(x)
  z_mean = Dense(hyperpars['latent_dim'], activation='linear')(x)
  z_std = Dense(hyperpars['latent_dim'], activation='softplus')(x)
  latents = Lambda(
      sample_gaussian, output_shape=(hyperpars['latent_dim'],), name='z')(
          [z_mean, z_std])
  encoder = Model(inputs=[image_inputs, label_inputs],
                  outputs=[z_mean, z_std, latents], name='encoder')
  
  # 2) Decoder
  latent_inputs = Input(shape=(hyperpars['latent_dim'],), name='latent_inputs')
  x = latent_inputs
  if hyperpars['conditional_vae']:
    x = Lambda(lambda x: K.concatenate(x, axis=-1))([x, label_inputs])
  x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(x)
  x = Reshape((shape[1], shape[2], shape[3]))(x)
  for (filters, kernel, strides) in hyperpars['filters_kernels_strides'][::-1]:
    x = Conv2DTranspose(filters=filters, kernel_size=kernel, strides=strides,
                        padding='same')(x)
  decoder_outputs = Conv2DTranspose(
      filters=1, kernel_size=3, activation='sigmoid', padding='same',
      name='decoder_outputs')(x)
  decoder = Model(inputs=[latent_inputs, label_inputs],
                  outputs=decoder_outputs, name='decoder')
  
  # 3) Combine the encoder and decoder into a Model with a custom loss.
  # Compute loss here because the VAE loss does not follow the standard format
  # See https://stackoverflow.com/questions/50063613/add-loss-function-in-keras
  # first decoder input = third encoder output
  outputs = decoder([ 
      encoder([image_inputs, label_inputs])[2], label_inputs]) 
  losses, metrics = vae_loss(z_mean, z_std, image_inputs, outputs, image_size,
                             hyperpars)  
  
  all_inputs = [image_inputs, label_inputs]
  model = Model(inputs=all_inputs, outputs=outputs, name='vae')
  model.add_loss(losses)
  
  return (model, encoder, decoder, metrics)


# Hack to add custom metrics when using add_loss
# Credit to May4m from https://github.com/keras-team/keras/issues/9459
def add_custom_metrics(model, custom_metrics):
  for k in custom_metrics:
    model.metrics_names.append(k)
    model.metrics_tensors.append(custom_metrics[k])


# Generate plots of the latent space, reconstructions and random samples.
def plot_results(models, data, latent_dim=2, conditional=False, batch_size=128,
                 model_name='vae_mnist_figures'):
  """Plots labels and MNIST digits as function of 2-dim latent vector."""
  encoder, decoder = models
  x_test, y_test, y_test_onehot = data
  os.makedirs(model_name, exist_ok=True)
  input_dims = 2 if conditional else 1

  # 1) Display a 2D plot of the digit classes in the latent space
  filename = os.path.join(model_name, "vae_mean.png")
  z_mean, _, _ = encoder.predict([x_test, y_test_onehot][:input_dims])
  plt.figure(figsize=(12, 10))
  plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
  plt.colorbar()
  plt.xlabel("z[0]")
  plt.ylabel("z[1]")
  plt.savefig(filename)
  plt.show()

  # 2) Display a 30x30 2D manifold of digits
  filename_base = os.path.join(model_name, "digits_over_latent")
  n = 30
  digit_size = 28
  # linearly spaced coordinates corresponding to the 2D plot
  # of digit classes in the latent space
  grid_x = np.linspace(-4, 4, n)
  grid_y = np.linspace(-4, 4, n)[::-1]

  # Loop over the latent dimensions: freeze all but 2 and sweep over these 2.
  num_latent_figures = latent_dim // 2
  for fig_id in range(num_latent_figures):
    figure = np.zeros((digit_size * n, digit_size * n))
    z_sample = np.random.normal(size=(1, latent_dim))
    z_digit = y_test_onehot[
        np.newaxis, np.random.randint(0, y_test.shape[0]), :]
    for i, yi in enumerate(grid_y):
      for j, xi in enumerate(grid_x):
        z_sample[0, int(fig_id*2)] = xi
        z_sample[0, int(fig_id*2 + 1)] = yi
        x_decoded = decoder.predict([z_sample, z_digit][:input_dims])
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit
  
    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename_base + str(fig_id) + '.png')
    plt.show()
    
  # 3) Display original images and reconstructions
  filename = os.path.join(model_name, "vae_reconstructions.png")
  num_reconstructions = 20
  sample_ids = np.random.choice(range(x_test.shape[0]), num_reconstructions,
                                      replace=False)
  sample_ids = sample_ids[y_test[sample_ids].argsort()]
  
  orig_images = x_test[sample_ids]
  orig_digits = y_test_onehot[sample_ids]
  _, _, encodings = encoder.predict([orig_images, orig_digits][:input_dims])
  reconstructions = decoder.predict([encodings, orig_digits][:input_dims])
  figure = np.zeros((digit_size*num_reconstructions, 2*digit_size))
  figure[:, :digit_size] = orig_images.reshape(-1, digit_size)
  figure[:, digit_size:] = reconstructions.reshape(-1, digit_size)
  plt.figure(figsize=(10, 10))
  plt.imshow(figure, cmap='Greys_r')
  plt.savefig(filename)
  plt.show()
  
  # 4) Generate random samples
  #    Keep the random latent samples fixed for all digits if using CVAE -
  #    this way you can inspect if the latent space is meaningful
  filename = os.path.join(model_name, "vae_generated_samples.png")
  random_samples_per_dim = 10
  random_codes = np.random.normal(size=(random_samples_per_dim, latent_dim))
  figure = np.zeros((random_samples_per_dim*digit_size,
                     random_samples_per_dim*digit_size))
  for i in range(random_samples_per_dim):
    if not conditional:
      random_codes = np.random.normal(
          size=(random_samples_per_dim, latent_dim))
    digit_onehot = to_categorical(
        np.repeat(np.array([i]), random_samples_per_dim), num_classes=10)
    reconstructions = decoder.predict([random_codes, digit_onehot][
        :input_dims])
    for j in range(random_samples_per_dim):
      figure[i*digit_size:((i+1)*digit_size),
             j*digit_size:((j+1)*digit_size)] = reconstructions[j, ..., 0]
  plt.figure(figsize=(random_samples_per_dim, random_samples_per_dim))
  plt.imshow(figure, cmap='Greys_r')
  plt.savefig(filename)
  plt.show()
  
  
# Custom Callback for checkpointing a specific model
# Inspired by https://stackoverflow.com/questions/50983008/how-to-save-best-weights-of-the-encoder-part-only-during-auto-encoder-training
# Callback source: https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L633
# Terrible BUG: the main model is saved when calling the second variable model.
class CustomCheckpointer(keras.callbacks.Callback):
  def __init__(self, filepath, custom_model, monitor, mode, save_best_only,
               verbose=0, verbose_description='encoder'):
    self.filepath = filepath
    self.custom_model = custom_model
    self.monitor = monitor
    self.save_best_only = save_best_only
    self.verbose = verbose
    self.description = verbose_description
    
    print('Initializing custom checkpointer for model `{}`.'.format(
        self.custom_model.name))
    self.monitor_op = np.less if mode == 'min' else np.greater
    self.best = np.Inf if mode == 'min' else -np.Inf
  
  def on_epoch_end(self, epoch, logs=None):
    current = logs.get(self.monitor)
    if not self.save_best_only or self.monitor_op(current, self.best):
      if self.verbose > 0:
        print('Saving the custom {} model to {}'.format(
            self.description, self.filepath))
      self.best = current
      self.custom_model.save(self.filepath, overwrite=True)
      
      
# Custom Keras callback for plotting learning progress
class PlotLosses(keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
    self.i = 0
    self.x = []
    self.val_losses = []
    self.fig = plt.figure()
    self.logs = []
    
    loss_extensions = ['', 'reconstruction', 'kl']
    self.best_loss_key = 'loss'
    self.loss_keys = [e + ('_' if e else '') + 'loss' for e in loss_extensions]
    self.losses = {k: [] for k in self.loss_keys}

  def on_epoch_end(self, epoch, logs={}):
    self.logs.append(logs)
    self.x.append(self.i)
    for k in self.loss_keys:
      self.losses[k].append(logs.get(k))
    self.i += 1
    
    best_loss = np.repeat(np.array(self.losses[self.best_loss_key]).min(),
                              self.i).tolist()
    best_id = (1+np.repeat(
        np.array(self.losses[self.best_loss_key]).argmin(), 2)).tolist()
    for k in self.loss_keys:
      plt.plot([1+x for x in self.x], self.losses[k], label=k)
    all_losses = np.array(list(self.losses.values())).flatten()
    if len(self.losses) > 1:
      plt.plot([1+x for x in self.x], best_loss, linestyle="--", color="r",
               label="")
      plt.plot(best_id, [0, best_loss[0]],
               linestyle="--", color="r", label="")
    plt.ylim(0, max(all_losses) + 0.1)
    plt.legend()
    plt.show()
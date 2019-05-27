# Basic GAN MNIST - utilities
# Inspired by https://medium.com/datadriveninvestor/generative-adversarial-network-gan-using-keras-ce1c05cfdfd3
# Loosely inspired by https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0
import keras
from keras import backend as K
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Reshape
from keras.models import Model
from keras.utils import to_categorical

import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

# Input data generator
def conditional_gan_generator(images, labels, hyperpars):
  batch_size = hyperpars['batch_size']
  latent_dim = hyperpars['latent_dim']
  true_data_target_prob = hyperpars['true_data_target_prob']
  half_batch_size = batch_size // 2
  steps_per_epoch=images.shape[0] // half_batch_size
  while True:
    shuffled_ids = np.random.permutation(images.shape[0])
    for i in range(steps_per_epoch):
      data_images = images[
          shuffled_ids[i*half_batch_size:((i+1)*half_batch_size)]]
      data_labels = labels[
          shuffled_ids[i*half_batch_size:((i+1)*half_batch_size)]]
      generated_noise = np.random.normal(size=(half_batch_size, latent_dim))
      generated_labels = to_categorical(np.random.randint(
          0, 10, size=(half_batch_size)), num_classes=10)
      is_true_im = np.ones((batch_size), dtype=float)*true_data_target_prob
      is_true_im[half_batch_size:] = 0
      
      # Two inputs, no outputs, flag if the epoch has been exhausted
      yield (data_images, data_labels, generated_noise, generated_labels,
             is_true_im, i==(steps_per_epoch-1))

# MNIST GAN model
def mnist_gan(hyperpars):
  image_size = 28
  image_inputs = Input((image_size, image_size, 1), name='image_inputs')
  label_inputs = Input((10,), dtype='float32', name='label_inputs')
  
  # 1) Discriminator
  x = image_inputs
  for (filters, kernel, strides) in hyperpars['filters_kernels_strides']:
    x = Conv2D(filters=filters, kernel_size=kernel, strides=strides,
               padding='same')(x)
  shape = K.int_shape(x) # shape info needed to build decoder model
  print('Shape after convolution: {}'.format(shape))
  x = Flatten()(x)
  if hyperpars['conditional_gan']:
    x = Lambda(lambda x: K.concatenate(x, axis=-1))([x, label_inputs])
  for layer_size in hyperpars['latent_mlp_layers']:
    x = Dense(layer_size, activation='relu')(x)
  prediction = Dense(1, activation='sigmoid')(x)
  discriminator = Model(inputs=[image_inputs, label_inputs],
                        outputs=[prediction], name='discriminator')
  
  # 2) Generator
  noise_inputs = Input(shape=(hyperpars['latent_dim'],), name='latent_inputs')
  x = noise_inputs
  if hyperpars['conditional_gan']:
    x = Lambda(lambda x: K.concatenate(x, axis=-1))([x, label_inputs])
  x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(x)
  x = Reshape((shape[1], shape[2], shape[3]))(x)
  for (filters, kernel, strides) in hyperpars['filters_kernels_strides'][::-1]:
    x = Conv2DTranspose(filters=filters, kernel_size=kernel, strides=strides,
                        padding='same')(x)
  generator_outputs = Conv2DTranspose(
      filters=1, kernel_size=3, activation='sigmoid', padding='same',
      name='generator_outputs')(x)
  generator = Model(inputs=[noise_inputs, label_inputs],
                    outputs=generator_outputs, name='generator')
  
  # 3) Chain the generator and discriminator to obtain a model that outputs
  #    a probability that the generated image is fake
  adversarial_prob_output = discriminator(
      [generator([noise_inputs, label_inputs]), label_inputs])
  adversarial_prob_model = Model(
      inputs=[noise_inputs, label_inputs],
      outputs=adversarial_prob_output, name='adversarial_prob_model')
  
  return (discriminator, generator, adversarial_prob_model)


def my_gan_fit(train_gen, discriminator, generator, adversarial_prob_model,
               hyperpars):
  num_epochs = hyperpars['num_epochs']
  half_batch_size = hyperpars['batch_size'] // 2
  batch_count = 60000 // half_batch_size
  for epoch_step in range(num_epochs):
    for batch_in_epoch in tqdm(range(batch_count)):
      (data_images, data_labels, generated_noise, generated_labels,
       is_true_im, end_of_epoch) = next(train_gen)
      
      # Train the discriminator with a single batch of data
      generated_images = generator.predict([generated_noise, generated_labels])
      images_concat = np.concatenate([data_images, generated_images], axis=0)
      labels_concat = np.concatenate([data_labels, generated_labels], axis=0)
      discriminator.trainable = True
      discr_loss = discriminator.train_on_batch(
          [images_concat, labels_concat], is_true_im)
      
      # Train the generator with a single batch of data
      discriminator.trainable = False
      gen_loss = adversarial_prob_model.train_on_batch(
          [generated_noise, generated_labels], np.ones(half_batch_size))
      
      if batch_in_epoch % 500 == 0:
        print(discr_loss, gen_loss)
      
    # End of epoch logic
    plot_generated_images(epoch_step, train_gen, generator, hyperpars)
    
    
def plot_generated_images(epoch_step, train_gen, generator, hyperpars,
                          model_name='gan_mnist_figures', digit_size=28):
  # Generate random samples
  # Keep the random latent samples fixed for all digits if using CVAE -
  # his way you can inspect if the latent space is meaningful
  filename = os.path.join(
      model_name, "gan_samples_epoch_" + str(epoch_step) + ".png")
  latent_dim = hyperpars['latent_dim'] 
  conditional = hyperpars['conditional_gan']
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
    samples = generator.predict([random_codes, digit_onehot])
    for j in range(random_samples_per_dim):
      figure[i*digit_size:((i+1)*digit_size),
             j*digit_size:((j+1)*digit_size)] = samples[j, ..., 0]
  plt.figure(figsize=(random_samples_per_dim, random_samples_per_dim))
  plt.imshow(figure, cmap='Greys_r')
  plt.savefig(filename)
  plt.show()


# Custom Callback for checkpointing a specific model
# Inspired by https://stackoverflow.com/questions/50983008/how-to-save-best-weights-of-the-encoder-part-only-during-auto-encoder-training
# Callback source: https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L633
# Terrible BUG: the main model is saved when calling the second variabel model.
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
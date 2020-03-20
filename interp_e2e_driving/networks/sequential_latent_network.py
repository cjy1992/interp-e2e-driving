# Copyright (c) 2020: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This file is modified from <https://github.com/alexlee-gk/slac>:
# Copyright (c) 2019 alexlee-gk
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import gin
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.trajectories import time_step as ts

from interp_e2e_driving.utils import nest_utils

tfd = tfp.distributions


class MultivariateNormalDiag(tf.Module):
  def __init__(self, base_depth, latent_size, name=None):
    super(MultivariateNormalDiag, self).__init__(name=name)
    self.latent_size = latent_size
    self.dense1 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
    self.dense2 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
    self.output_layer = tf.keras.layers.Dense(2 * latent_size)

  def __call__(self, *inputs):
    if len(inputs) > 1:
      inputs = tf.concat(inputs, axis=-1)
    else:
      inputs, = inputs
    out = self.dense1(inputs)
    out = self.dense2(out)
    out = self.output_layer(out)
    loc = out[..., :self.latent_size]
    scale_diag = tf.nn.softplus(out[..., self.latent_size:]) + 1e-5
    return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)


class ConstantMultivariateNormalDiag(tf.Module):
  def __init__(self, latent_size, name=None):
    super(ConstantMultivariateNormalDiag, self).__init__(name=name)
    self.latent_size = latent_size

  def __call__(self, *inputs):
    # first input should not have any dimensions after the batch_shape, step_type
    batch_shape = tf.shape(inputs[0])  # input is only used to infer batch_shape
    shape = tf.concat([batch_shape, [self.latent_size]], axis=0)
    loc = tf.zeros(shape)
    scale_diag = tf.ones(shape)
    return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)


class Decoder64(tf.Module):
  """Probabilistic decoder for `p(x_t | z_t)`, with input image size of 64*64.
  """

  def __init__(self, base_depth, channels=3, scale=1.0, name=None):
    super(Decoder64, self).__init__(name=name)
    self.scale = scale
    conv_transpose = functools.partial(
        tf.keras.layers.Conv2DTranspose, padding="SAME", activation=tf.nn.leaky_relu)
    self.conv_transpose1 = conv_transpose(8 * base_depth, 4, padding="VALID")
    self.conv_transpose2 = conv_transpose(4 * base_depth, 3, 2)
    self.conv_transpose3 = conv_transpose(2 * base_depth, 3, 2)
    self.conv_transpose4 = conv_transpose(base_depth, 3, 2)
    self.conv_transpose5 = conv_transpose(channels, 5, 2)

  def __call__(self, *inputs):
    if len(inputs) > 1:
      latent = tf.concat(inputs, axis=-1)
    else:
      latent, = inputs
    # (sample, N, T, latent)
    collapsed_shape = tf.stack([-1, 1, 1, tf.shape(latent)[-1]], axis=0)
    out = tf.reshape(latent, collapsed_shape)
    out = self.conv_transpose1(out)
    out = self.conv_transpose2(out)
    out = self.conv_transpose3(out)
    out = self.conv_transpose4(out)
    out = self.conv_transpose5(out)  # (sample*N*T, h, w, c)

    expanded_shape = tf.concat(
        [tf.shape(latent)[:-1], tf.shape(out)[1:]], axis=0)
    out = tf.reshape(out, expanded_shape)  # (sample, N, T, h, w, c)
    return tfd.Independent(
        distribution=tfd.Normal(loc=out, scale=self.scale),
        reinterpreted_batch_ndims=3)  # wrap (h, w, c)


class Decoder128(tf.Module):
  """Probabilistic decoder for `p(x_t | z_t)`, with input image size of 128*128.
  """

  def __init__(self, base_depth, channels=3, scale=1.0, name=None):
    super(Decoder128, self).__init__(name=name)
    self.scale = scale
    conv_transpose = functools.partial(
        tf.keras.layers.Conv2DTranspose, padding="SAME", activation=tf.nn.leaky_relu)
    self.conv_transpose1 = conv_transpose(8 * base_depth, 4, padding="VALID")
    self.conv_transpose2 = conv_transpose(8 * base_depth, 3, 2)
    self.conv_transpose3 = conv_transpose(4 * base_depth, 3, 2)
    self.conv_transpose4 = conv_transpose(2 * base_depth, 3, 2)
    self.conv_transpose5 = conv_transpose(base_depth, 3, 2)
    self.conv_transpose6 = conv_transpose(channels, 5, 2)

  def __call__(self, *inputs):
    if len(inputs) > 1:
      latent = tf.concat(inputs, axis=-1)
    else:
      latent, = inputs
    # (sample, N, T, latent)
    collapsed_shape = tf.stack([-1, 1, 1, tf.shape(latent)[-1]], axis=0)
    out = tf.reshape(latent, collapsed_shape)
    out = self.conv_transpose1(out)
    out = self.conv_transpose2(out)
    out = self.conv_transpose3(out)
    out = self.conv_transpose4(out)
    out = self.conv_transpose5(out)
    out = self.conv_transpose6(out)  # (sample*N*T, h, w, c)

    expanded_shape = tf.concat(
        [tf.shape(latent)[:-1], tf.shape(out)[1:]], axis=0)
    out = tf.reshape(out, expanded_shape)  # (sample, N, T, h, w, c)
    return tfd.Independent(
        distribution=tfd.Normal(loc=out, scale=self.scale),
        reinterpreted_batch_ndims=3)  # wrap (h, w, c)


class Decoder256(tf.Module):
  """Probabilistic decoder for `p(x_t | z_t)`, with input image size of 256*256.
  """

  def __init__(self, base_depth, channels=3, scale=1.0, name=None):
    super(Decoder256, self).__init__(name=name)
    self.scale = scale
    conv_transpose = functools.partial(
        tf.keras.layers.Conv2DTranspose, padding="SAME", activation=tf.nn.leaky_relu)
    self.conv_transpose1 = conv_transpose(8 * base_depth, 4, padding="VALID")
    self.conv_transpose2 = conv_transpose(8 * base_depth, 3, 2)
    self.conv_transpose3 = conv_transpose(8 * base_depth, 3, 2)
    self.conv_transpose4 = conv_transpose(4 * base_depth, 3, 2)
    self.conv_transpose5 = conv_transpose(2 * base_depth, 3, 2)
    self.conv_transpose6 = conv_transpose(base_depth, 3, 2)
    self.conv_transpose7 = conv_transpose(channels, 5, 2)

  def __call__(self, *inputs):
    if len(inputs) > 1:
      latent = tf.concat(inputs, axis=-1)
    else:
      latent, = inputs
    # (sample, N, T, latent)
    collapsed_shape = tf.stack([-1, 1, 1, tf.shape(latent)[-1]], axis=0)
    out = tf.reshape(latent, collapsed_shape)
    out = self.conv_transpose1(out)
    out = self.conv_transpose2(out)
    out = self.conv_transpose3(out)
    out = self.conv_transpose4(out)
    out = self.conv_transpose5(out)
    out = self.conv_transpose6(out)
    out = self.conv_transpose7(out)  # (sample*N*T, h, w, c)

    expanded_shape = tf.concat(
        [tf.shape(latent)[:-1], tf.shape(out)[1:]], axis=0)
    out = tf.reshape(out, expanded_shape)  # (sample, N, T, h, w, c)
    return tfd.Independent(
        distribution=tfd.Normal(loc=out, scale=self.scale),
        reinterpreted_batch_ndims=3)  # wrap (h, w, c)


class Encoder64(tf.Module):
  """Feature extractor for input image size 64*64.
  """

  def __init__(self, base_depth, feature_size, name=None):
    super(Encoder64, self).__init__(name=name)
    self.feature_size = feature_size
    conv = functools.partial(
        tf.keras.layers.Conv2D, padding="SAME", activation=tf.nn.leaky_relu)
    self.conv1 = conv(base_depth, 5, 2)
    self.conv2 = conv(2 * base_depth, 3, 2)
    self.conv3 = conv(4 * base_depth, 3, 2)
    self.conv4 = conv(8 * base_depth, 3, 2)
    self.conv5 = conv(8 * base_depth, 4, padding="VALID")

  def __call__(self, image):
    image_shape = tf.shape(image)[-3:]
    collapsed_shape = tf.concat(([-1], image_shape), axis=0)
    out = tf.reshape(image, collapsed_shape)  # (sample*N*T, h, w, c)
    out = self.conv1(out)
    out = self.conv2(out)
    out = self.conv3(out)
    out = self.conv4(out)
    out = self.conv5(out)
    expanded_shape = tf.concat((tf.shape(image)[:-3], [self.feature_size]), axis=0)
    return tf.reshape(out, expanded_shape)  # (sample, N, T, feature)


class Encoder128(tf.Module):
  """Feature extractor for input image size 128*128.
  """

  def __init__(self, base_depth, feature_size, name=None):
    super(Encoder128, self).__init__(name=name)
    self.feature_size = feature_size
    conv = functools.partial(
        tf.keras.layers.Conv2D, padding="SAME", activation=tf.nn.leaky_relu)
    self.conv1 = conv(base_depth, 5, 2)
    self.conv2 = conv(2 * base_depth, 3, 2)
    self.conv3 = conv(4 * base_depth, 3, 2)
    self.conv4 = conv(8 * base_depth, 3, 2)
    self.conv5 = conv(8 * base_depth, 3, 2)
    self.conv6 = conv(8 * base_depth, 4, padding="VALID")

  def __call__(self, image):
    image_shape = tf.shape(image)[-3:]
    collapsed_shape = tf.concat(([-1], image_shape), axis=0)
    out = tf.reshape(image, collapsed_shape)  # (sample*N*T, h, w, c)
    out = self.conv1(out)
    out = self.conv2(out)
    out = self.conv3(out)
    out = self.conv4(out)
    out = self.conv5(out)
    out = self.conv6(out)
    expanded_shape = tf.concat((tf.shape(image)[:-3], [self.feature_size]), axis=0)
    return tf.reshape(out, expanded_shape)  # (sample, N, T, feature)


class Encoder256(tf.Module):
  """Feature extractor for input image size 256*256.
  """

  def __init__(self, base_depth, feature_size, name=None):
    super(Encoder256, self).__init__(name=name)
    self.feature_size = feature_size
    conv = functools.partial(
        tf.keras.layers.Conv2D, padding="SAME", activation=tf.nn.leaky_relu)
    self.conv1 = conv(base_depth, 5, 2)
    self.conv2 = conv(2 * base_depth, 3, 2)
    self.conv3 = conv(4 * base_depth, 3, 2)
    self.conv4 = conv(8 * base_depth, 3, 2)
    self.conv5 = conv(8 * base_depth, 3, 2)
    self.conv6 = conv(8 * base_depth, 3, 2)
    self.conv7 = conv(8 * base_depth, 4, padding="VALID")

  def __call__(self, image):
    image_shape = tf.shape(image)[-3:]
    collapsed_shape = tf.concat(([-1], image_shape), axis=0)
    out = tf.reshape(image, collapsed_shape)  # (sample*N*T, h, w, c)
    out = self.conv1(out)
    out = self.conv2(out)
    out = self.conv3(out)
    out = self.conv4(out)
    out = self.conv5(out)
    out = self.conv6(out)
    out = self.conv7(out)
    expanded_shape = tf.concat((tf.shape(image)[:-3], [self.feature_size]), axis=0)
    return tf.reshape(out, expanded_shape)  # (sample, N, T, feature)


@gin.configurable
class SequentialLatentModelHierarchical(tf.Module):
  """The hierarchical sequential latent model. See https://arxiv.org/abs/1907.00953.
  """

  def __init__(self,
               input_names,
               reconstruct_names,
               obs_size=64,
               base_depth=32,
               latent1_size=32,
               latent2_size=256,
               kl_analytic=True,
               decoder_stddev=np.sqrt(0.1, dtype=np.float32),
               name=None):
    """Creates an instance of `SequentialLatentModelHierarchical`.
    Args:
      input_names: the names of the observation inputs (e.g, 'camera', 'lidar').
      reconstruct_names: names of the outputs to reconstruct (e.g, 'mask').
      obs_size: the pixel size of the observation inputs. Here we assume
        the image inputs have same width and height.
      base_depth: base depth of the convolutional layers.
      latent1_size: size of the first latent of the hierarchical latent model.
      latent2_size: size of the second latent of the hierarchical latent model.
      kl_analytic: whether to use analytical KL divergence.
      decoder_stddev: standard deviation of the decoder.
      name: A string representing name of the network.
    """
    super(SequentialLatentModelHierarchical, self).__init__(name=name)
    self.input_names = input_names
    self.reconstruct_names = reconstruct_names
    self.latent1_size = latent1_size
    self.latent2_size = latent2_size
    self.kl_analytic = kl_analytic
    self.obs_size = obs_size

    latent1_first_prior_distribution_ctor = ConstantMultivariateNormalDiag
    latent1_distribution_ctor = MultivariateNormalDiag
    latent2_distribution_ctor = MultivariateNormalDiag

    # p(z_1^1)
    self.latent1_first_prior = latent1_first_prior_distribution_ctor(latent1_size)
    # p(z_1^2 | z_1^1)
    self.latent2_first_prior = latent2_distribution_ctor(8 * base_depth, latent2_size)
    # p(z_{t+1}^1 | z_t^2, a_t)
    self.latent1_prior = latent1_distribution_ctor(8 * base_depth, latent1_size)
    # p(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t)
    self.latent2_prior = latent2_distribution_ctor(8 * base_depth, latent2_size)

    # q(z_1^1 | f_1)
    self.latent1_first_posterior = latent1_distribution_ctor(8 * base_depth, latent1_size)
    # q(z_1^2 | z_1^1) = p(z_1^2 | z_1^1)
    self.latent2_first_posterior = self.latent2_first_prior
    # q(z_{t+1}^1 | f_{t+1}, z_t^2, a_t)
    self.latent1_posterior = latent1_distribution_ctor(8 * base_depth, latent1_size)
    # q(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t) = p(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t)
    self.latent2_posterior = self.latent2_prior

    # Create encoders q(f_t|x_t)
    self.encoders = {}
    for name in input_names:
      if obs_size == 64:
        self.encoders[name] = Encoder64(base_depth, 8 * base_depth)
      elif obs_size == 128:
        self.encoders[name] = Encoder128(base_depth, 8 * base_depth)
      elif obs_size == 256:
        self.encoders[name] = Encoder256(base_depth, 8 * base_depth)
      else:
        raise NotImplementedError

    # Create decoders q(x_t|z_t)
    self.decoders = {}
    for name in self.reconstruct_names:
      if obs_size == 64:
        self.decoders[name] = Decoder64(base_depth, scale=decoder_stddev)
      elif obs_size == 128:
        self.decoders[name] = Decoder128(base_depth, scale=decoder_stddev)
      elif obs_size == 256:
        self.decoders[name] = Decoder256(base_depth, scale=decoder_stddev)
      else:
        raise NotImplementedError

  @property
  def latent_size(self):
    """Size of the latent state."""
    return self.latent1_size + self.latent2_size

  def sample_prior(self, batch_size):
    """Sample the prior latent state."""
    latent1 = self.latent1_first_prior(tf.zeros(batch_size)).sample()
    latent2 = self.latent2_first_prior(latent1).sample()
    latent = tf.concat([latent1, latent2], axis=-1)
    return latent

  def filter(self, image, last_latent, last_action):
    """Apply recursive filter to obtain posterior estimation of latent 
      q(z_{t+1}|z_t,a_t,x_{t+1}).
    """
    feature = self.get_features(image)
    last_latent2 = tf.gather(last_latent, tf.range(self.latent1_size, self.latent_size), axis=-1)
    latent1 = self.latent1_posterior(feature, last_latent2, last_action).sample()
    latent2 = self.latent2_posterior(latent1, last_latent2, last_action).sample()
    latent = tf.concat([latent1, latent2], axis=-1)
    return latent

  def first_filter(self, image):
    """Obtain the posterior of the latent at the first timestep q(z_1|x_1)."""
    feature = self.get_features(image)
    latent1 = self.latent1_first_posterior(feature).sample()
    latent2 = self.latent2_first_posterior(latent1).sample()
    latent = tf.concat([latent1, latent2], axis=-1)
    return latent

  def get_features(self, images):
    """Get low dimensional features from images q(f_t|x_t)"""
    features = {}
    for name in self.input_names:
      images_tmp = tf.image.convert_image_dtype(images[name], tf.float32)
      features[name] = self.encoders[name](images_tmp)
    features = sum(features.values())
    return features

  def reconstruct(self, latent):
    """Reconstruct the images in reconstruct_names given the latent state."""
    latent1 = tf.gather(latent, tf.range(0, self.latent1_size), axis=-1)
    latent2 = tf.gather(latent, tf.range(self.latent1_size, self.latent_size), axis=-1)
    posterior_images = {}
    for name in self.reconstruct_names:
      posterior_images[name] = self.decoders[name](latent1, latent2).mean()
    posterior_images = tf.concat(list(posterior_images.values()), axis=-2)
    return posterior_images

  def compute_latents(self, images, actions, step_types, latent_posterior_samples_and_dists=None, num_first_image=5):
    """Compute the latent states of the sequential latent model."""
    sequence_length = step_types.shape[1] - 1

    # Get posterior and prior samples of latents
    if latent_posterior_samples_and_dists is None:
      latent_posterior_samples_and_dists = self.sample_posterior(images, actions, step_types)
    (latent1_posterior_samples, latent2_posterior_samples), (latent1_posterior_dists, latent2_posterior_dists) = (
        latent_posterior_samples_and_dists)
    (latent1_prior_samples, latent2_prior_samples), _ = self.sample_prior_or_posterior(actions, step_types)  # for visualization

    # Get prior samples of latents conditioned on intial inputs
    first_image = {}
    for k,v in images.items():
      first_image[k] = v[:, :num_first_image]
    (latent1_conditional_prior_samples, latent2_conditional_prior_samples), _ = self.sample_prior_or_posterior(
        actions, step_types, images=first_image)  # for visualization. condition on first image only

    # Reset the initial steps of an episode to first prior latents
    def where_and_concat(reset_masks, first_prior_tensors, after_first_prior_tensors):
      after_first_prior_tensors = tf.where(reset_masks[:, 1:], first_prior_tensors[:, 1:], after_first_prior_tensors)
      prior_tensors = tf.concat([first_prior_tensors[:, 0:1], after_first_prior_tensors], axis=1)
      return prior_tensors

    reset_masks = tf.concat([tf.ones_like(step_types[:, 0:1], dtype=tf.bool),
                             tf.equal(step_types[:, 1:], ts.StepType.FIRST)], axis=1)

    latent1_reset_masks = tf.tile(reset_masks[:, :, None], [1, 1, self.latent1_size])
    latent1_first_prior_dists = self.latent1_first_prior(step_types)
    latent1_after_first_prior_dists = self.latent1_prior(
        latent2_posterior_samples[:, :sequence_length], actions[:, :sequence_length])
    latent1_prior_dists = nest_utils.map_distribution_structure(
        functools.partial(where_and_concat, latent1_reset_masks),
        latent1_first_prior_dists,
        latent1_after_first_prior_dists)

    latent2_reset_masks = tf.tile(reset_masks[:, :, None], [1, 1, self.latent2_size])
    latent2_first_prior_dists = self.latent2_first_prior(latent1_posterior_samples)
    latent2_after_first_prior_dists = self.latent2_prior(
        latent1_posterior_samples[:, 1:sequence_length+1],
        latent2_posterior_samples[:, :sequence_length],
        actions[:, :sequence_length])
    latent2_prior_dists = nest_utils.map_distribution_structure(
        functools.partial(where_and_concat, latent2_reset_masks),
        latent2_first_prior_dists,
        latent2_after_first_prior_dists)

    return (latent1_posterior_dists, latent1_prior_dists), (latent2_posterior_dists,
      latent2_prior_dists), (latent1_posterior_samples, latent1_prior_samples,
      latent1_conditional_prior_samples), (latent2_posterior_samples, latent2_prior_samples,
      latent2_conditional_prior_samples)


  def compute_loss(self, images, actions, step_types, latent_posterior_samples_and_dists=None, num_first_image=5):
    # Compuate the latents
    latent1_dists, latent2_dists, latent1_samples, latent2_samples = \
      self.compute_latents(images, actions, step_types, latent_posterior_samples_and_dists, num_first_image)

    latent1_posterior_dists, latent1_prior_dists = latent1_dists
    latent2_posterior_dists, latent2_prior_dists = latent2_dists
    latent1_posterior_samples, latent1_prior_samples, \
      latent1_conditional_prior_samples = latent1_samples
    latent2_posterior_samples, latent2_prior_samples, \
      latent2_conditional_prior_samples = latent2_samples

    # Compute the KL divergence part of the ELBO
    outputs = {}
    if self.kl_analytic:
      latent1_kl_divergences = tfd.kl_divergence(latent1_posterior_dists, latent1_prior_dists)
    else:
      latent1_kl_divergences = (latent1_posterior_dists.log_prob(latent1_posterior_samples)
                                - latent1_prior_dists.log_prob(latent1_posterior_samples))
    latent1_kl_divergences = tf.reduce_sum(latent1_kl_divergences, axis=1)
    outputs.update({
      'latent1_kl_divergence': tf.reduce_mean(latent1_kl_divergences),
    })

    if self.kl_analytic:
      latent2_kl_divergences = tfd.kl_divergence(latent2_posterior_dists, latent2_prior_dists)
    else:
      latent2_kl_divergences = (latent2_posterior_dists.log_prob(latent2_posterior_samples)
                                - latent2_prior_dists.log_prob(latent2_posterior_samples))
    latent2_kl_divergences = tf.reduce_sum(latent2_kl_divergences, axis=1)
    outputs.update({
      'latent2_kl_divergence': tf.reduce_mean(latent2_kl_divergences),
    })

    outputs.update({
      'kl_divergence': tf.reduce_mean(latent1_kl_divergences + latent2_kl_divergences),
    })

    elbo = - latent1_kl_divergences - latent2_kl_divergences

    # Compute the reconstruction part of the ELBO
    likelihood_dists = {}
    likelihood_log_probs = {}
    reconstruction_error = {}
    for name in self.reconstruct_names:
      likelihood_dists[name] = self.decoders[name](latent1_posterior_samples, latent2_posterior_samples)
      images_tmp = tf.image.convert_image_dtype(images[name], tf.float32)
      likelihood_log_probs[name] = likelihood_dists[name].log_prob(images_tmp)
      likelihood_log_probs[name] = tf.reduce_sum(likelihood_log_probs[name], axis=1)
      reconstruction_error[name] = tf.reduce_sum(tf.square(images_tmp - likelihood_dists[name].distribution.loc),
                                         axis=list(range(-len(likelihood_dists[name].event_shape), 0)))
      reconstruction_error[name] = tf.reduce_sum(reconstruction_error[name], axis=1)
      outputs.update({
        'log_likelihood_'+name: tf.reduce_mean(likelihood_log_probs[name]),
        'reconstruction_error_'+name: tf.reduce_mean(reconstruction_error[name]),
      })
      elbo += likelihood_log_probs[name]

    # Compute the loss
    loss = -tf.reduce_mean(elbo)

    # Generate the images
    posterior_images = {}
    prior_images = {}
    conditional_prior_images = {}
    for name in self.reconstruct_names:
      posterior_images[name] = likelihood_dists[name].mean()
      prior_images[name] = self.decoders[name](latent1_prior_samples, latent2_prior_samples).mean()
      conditional_prior_images[name] = self.decoders[name](latent1_conditional_prior_samples, latent2_conditional_prior_samples).mean()

    images = tf.concat([tf.image.convert_image_dtype(images[k], tf.float32)
      for k in list(set(self.input_names+self.reconstruct_names))], axis=-2)
    posterior_images = tf.concat(list(posterior_images.values()), axis=-2)
    prior_images = tf.concat(list(prior_images.values()), axis=-2)
    conditional_prior_images = tf.concat(list(conditional_prior_images.values()), axis=-2)

    outputs.update({
      'elbo': tf.reduce_mean(elbo),
      'images': images,
      'posterior_images': posterior_images,
      'prior_images': prior_images,
      'conditional_prior_images': conditional_prior_images,
    })
    return loss, outputs

  def sample_prior_or_posterior(self, actions, step_types=None, images=None):
    """Samples from the prior latents, or latents conditioned on input images."""
    if step_types is None:
      batch_size = tf.shape(actions)[0]
      sequence_length = actions.shape[1]  # should be statically defined
      step_types = tf.fill(
          [batch_size, sequence_length + 1], ts.StepType.MID)
    else:
      sequence_length = step_types.shape[1] - 1
      actions = actions[:, :sequence_length]
    if images is not None:
      features = self.get_features(images)

    # Swap batch and time axes
    actions = tf.transpose(actions, [1, 0, 2])
    step_types = tf.transpose(step_types, [1, 0])
    if images is not None:
      features = tf.transpose(features, [1, 0, 2])

    # Get latent distributions and samples
    latent1_dists = []
    latent1_samples = []
    latent2_dists = []
    latent2_samples = []
    for t in range(sequence_length + 1):
      is_conditional = images is not None and (t < list(images.values())[0].shape[1])
      if t == 0:
        if is_conditional:
          latent1_dist = self.latent1_first_posterior(features[t])
        else:
          latent1_dist = self.latent1_first_prior(step_types[t])  # step_types is only used to infer batch_size
        latent1_sample = latent1_dist.sample()
        if is_conditional:
          latent2_dist = self.latent2_first_posterior(latent1_sample)
        else:
          latent2_dist = self.latent2_first_prior(latent1_sample)
        latent2_sample = latent2_dist.sample()
      else:
        reset_mask = tf.equal(step_types[t], ts.StepType.FIRST)
        reset_mask = tf.expand_dims(reset_mask, 1)
        if is_conditional:
          latent1_first_dist = self.latent1_first_posterior(features[t])
          latent1_dist = self.latent1_posterior(features[t], latent2_samples[t-1], actions[t-1])
        else:
          latent1_first_dist = self.latent1_first_prior(step_types[t])
          latent1_dist = self.latent1_prior(latent2_samples[t-1], actions[t-1])
        latent1_dist = nest_utils.map_distribution_structure(
            functools.partial(tf.where, reset_mask), latent1_first_dist, latent1_dist)
        latent1_sample = latent1_dist.sample()

        if is_conditional:
          latent2_first_dist = self.latent2_first_posterior(latent1_sample)
          latent2_dist = self.latent2_posterior(latent1_sample, latent2_samples[t-1], actions[t-1])
        else:
          latent2_first_dist = self.latent2_first_prior(latent1_sample)
          latent2_dist = self.latent2_prior(latent1_sample, latent2_samples[t-1], actions[t-1])
        latent2_dist = nest_utils.map_distribution_structure(
            functools.partial(tf.where, reset_mask), latent2_first_dist, latent2_dist)
        latent2_sample = latent2_dist.sample()

      latent1_dists.append(latent1_dist)
      latent1_samples.append(latent1_sample)
      latent2_dists.append(latent2_dist)
      latent2_samples.append(latent2_sample)

    latent1_dists = nest_utils.map_distribution_structure(lambda *x: tf.stack(x, axis=1), *latent1_dists)
    latent1_samples = tf.stack(latent1_samples, axis=1)
    latent2_dists = nest_utils.map_distribution_structure(lambda *x: tf.stack(x, axis=1), *latent2_dists)
    latent2_samples = tf.stack(latent2_samples, axis=1)
    return (latent1_samples, latent2_samples), (latent1_dists, latent2_dists)

  def sample_posterior(self, images, actions, step_types, features=None):
    """Sample posterior latents conditioned on input images."""
    sequence_length = step_types.shape[1] - 1
    actions = actions[:, :sequence_length]

    if features is None:
      features = self.get_features(images)

    # Swap batch and time axes
    features = tf.transpose(features, [1, 0, 2])
    actions = tf.transpose(actions, [1, 0, 2])
    step_types = tf.transpose(step_types, [1, 0])

    # Get latent distributions and samples
    latent1_dists = []
    latent1_samples = []
    latent2_dists = []
    latent2_samples = []
    for t in range(sequence_length + 1):
      if t == 0:
        latent1_dist = self.latent1_first_posterior(features[t])
        latent1_sample = latent1_dist.sample()
        latent2_dist = self.latent2_first_posterior(latent1_sample)
        latent2_sample = latent2_dist.sample()
      else:
        reset_mask = tf.equal(step_types[t], ts.StepType.FIRST)
        reset_mask = tf.expand_dims(reset_mask, 1)
        latent1_first_dist = self.latent1_first_posterior(features[t])
        latent1_dist = self.latent1_posterior(features[t], latent2_samples[t-1], actions[t-1])
        latent1_dist = nest_utils.map_distribution_structure(
            functools.partial(tf.where, reset_mask), latent1_first_dist, latent1_dist)
        latent1_sample = latent1_dist.sample()

        latent2_first_dist = self.latent2_first_posterior(latent1_sample)
        latent2_dist = self.latent2_posterior(latent1_sample, latent2_samples[t-1], actions[t-1])
        latent2_dist = nest_utils.map_distribution_structure(
            functools.partial(tf.where, reset_mask), latent2_first_dist, latent2_dist)
        latent2_sample = latent2_dist.sample()

      latent1_dists.append(latent1_dist)
      latent1_samples.append(latent1_sample)
      latent2_dists.append(latent2_dist)
      latent2_samples.append(latent2_sample)

    latent1_dists = nest_utils.map_distribution_structure(lambda *x: tf.stack(x, axis=1), *latent1_dists)
    latent1_samples = tf.stack(latent1_samples, axis=1)
    latent2_dists = nest_utils.map_distribution_structure(lambda *x: tf.stack(x, axis=1), *latent2_dists)
    latent2_samples = tf.stack(latent2_samples, axis=1)
    return (latent1_samples, latent2_samples), (latent1_dists, latent2_dists)


@gin.configurable
class SequentialLatentModelNonHierarchical(tf.Module):

  def __init__(self,
               input_names,
               reconstruct_names,
               obs_size=64,
               base_depth=32,
               latent_size=64,
               kl_analytic=True,
               decoder_stddev=np.sqrt(0.1, dtype=np.float32),
               name=None):
    """Creates an instance of `SequentialLatentModelHierarchical`.
    Args:
      input_names: the names of the observation inputs (e.g, 'camera', 'lidar').
      reconstruct_names: names of the outputs to reconstruct (e.g, 'mask').
      obs_size: the pixel size of the observation inputs. Here we assume
        the image inputs have same width and height.
      base_depth: base depth of the convolutional layers.
      latent_size: size of the latent state.
      kl_analytic: whether to use analytical KL divergence.
      decoder_stddev: standard deviation of the decoder.
      name: A string representing name of the network.
    """
    super(SequentialLatentModelNonHierarchical, self).__init__(name=name)
    self.input_names = input_names
    self.reconstruct_names = reconstruct_names
    self.latent_size = latent_size
    self.kl_analytic = kl_analytic
    self.obs_size = obs_size

    latent_first_prior_distribution_ctor = ConstantMultivariateNormalDiag
    latent_distribution_ctor = MultivariateNormalDiag

    # p(z_1)
    self.latent_first_prior = latent_first_prior_distribution_ctor(latent_size)
    # p(z_{t+1} | z_t, a_t)
    self.latent_prior = latent_distribution_ctor(8 * base_depth, latent_size)

    # q(z_1 | f_1)
    self.latent_first_posterior = latent_distribution_ctor(8 * base_depth, latent_size)
    # q(z_{t+1} | f_{t+1}, z_t, a_t)
    self.latent_posterior = latent_distribution_ctor(8 * base_depth, latent_size)

    # Create encoders q(f_t|x_t)
    self.encoders = {}
    for name in input_names:
      if obs_size == 64:
        self.encoders[name] = Encoder64(base_depth, 8 * base_depth)
      elif obs_size == 128:
        self.encoders[name] = Encoder128(base_depth, 8 * base_depth)
      elif obs_size == 256:
        self.encoders[name] = Encoder256(base_depth, 8 * base_depth)
      else:
        raise NotImplementedError

    # Create decoders q(x_t|z_t)
    self.decoders = {}
    for name in self.reconstruct_names:
      if obs_size == 64:
        self.decoders[name] = Decoder64(base_depth, scale=decoder_stddev)
      elif obs_size == 128:
        self.decoders[name] = Decoder128(base_depth, scale=decoder_stddev)
      elif obs_size == 256:
        self.decoders[name] = Decoder256(base_depth, scale=decoder_stddev)
      else:
        raise NotImplementedError

  def sample_prior(self, batch_size):
    """Sample the prior latent state."""
    latent = self.latent_first_prior(tf.zeros(batch_size)).sample()
    return latent

  def filter(self, image, last_latent, last_action):
    """Apply recursive filter to obtain posterior estimation of latent 
      q(z_{t+1}|z_t,a_t,x_{t+1}).
    """
    feature = self.get_features(image)
    latent = self.latent_posterior(feature, last_latent, last_action).sample()
    return latent

  def first_filter(self, image):
    """Obtain the posterior of the latent at the first timestep q(z_1|x_1)."""
    feature = self.get_features(image)
    latent = self.latent_first_posterior(feature).sample()
    return latent

  def get_features(self, images):
    """Get low dimensional features from images q(f_t|x_t)"""
    features = {}
    for name in self.input_names:
      images_tmp = tf.image.convert_image_dtype(images[name], tf.float32)
      features[name] = self.encoders[name](images_tmp)
    features = sum(features.values())
    return features

  def reconstruct(self, latent):
    """Reconstruct the images in reconstruct_names given the latent state."""
    posterior_images = {}
    for name in self.reconstruct_names:
      posterior_images[name] = self.decoders[name](latent).mean()
    posterior_images = tf.concat(list(posterior_images.values()), axis=-2)
    return posterior_images

  def compute_latents(self, images, actions, step_types, latent_posterior_samples_and_dists=None):
    """Compute the latent states of the sequential latent model."""
    sequence_length = step_types.shape[1] - 1

    # Get posterior and prior samples of latents
    if latent_posterior_samples_and_dists is None:
      latent_posterior_samples_and_dists = self.sample_posterior(images, actions, step_types)
    latent_posterior_samples, latent_posterior_dists = latent_posterior_samples_and_dists
    latent_prior_samples, _ = self.sample_prior_or_posterior(actions, step_types)  # for visualization

    # Get prior samples of latents conditioned on intial inputs
    first_image = {}
    num_first_image = 3
    for k,v in images.items():
      first_image[k] = v[:, :num_first_image]
    latent_conditional_prior_samples, _ = self.sample_prior_or_posterior(
        actions, step_types, images=first_image)  # for visualization. condition on first image only

    # Reset the initial steps of an episode to first prior latents
    def where_and_concat(reset_masks, first_prior_tensors, after_first_prior_tensors):
      after_first_prior_tensors = tf.where(reset_masks[:, 1:], first_prior_tensors[:, 1:], after_first_prior_tensors)
      prior_tensors = tf.concat([first_prior_tensors[:, 0:1], after_first_prior_tensors], axis=1)
      return prior_tensors

    reset_masks = tf.concat([tf.ones_like(step_types[:, 0:1], dtype=tf.bool),
                             tf.equal(step_types[:, 1:], ts.StepType.FIRST)], axis=1)

    latent_reset_masks = tf.tile(reset_masks[:, :, None], [1, 1, self.latent_size])
    latent_first_prior_dists = self.latent_first_prior(step_types)
    # these distributions start at t=1 and the inputs are from t-1
    latent_after_first_prior_dists = self.latent_prior(
        latent_posterior_samples[:, :sequence_length], actions[:, :sequence_length])
    latent_prior_dists = nest_utils.map_distribution_structure(
        functools.partial(where_and_concat, latent_reset_masks),
        latent_first_prior_dists,
        latent_after_first_prior_dists)

    return (latent_posterior_dists, latent_prior_dists), (latent_posterior_samples,
      latent_prior_samples, latent_conditional_prior_samples)

  def compute_loss(self, images, actions, step_types, latent_posterior_samples_and_dists=None):
    # Compuate the latents
    latent_dists, latent_samples = self.compute_latents(images, actions, step_types, latent_posterior_samples_and_dists)
    latent_posterior_dists, latent_prior_dists = latent_dists
    latent_posterior_samples, latent_prior_samples, latent_conditional_prior_samples = latent_samples

    # Compute the KL divergence part of the ELBO
    outputs = {}
    if self.kl_analytic:
      latent_kl_divergences = tfd.kl_divergence(latent_posterior_dists, latent_prior_dists)
    else:
      latent_kl_divergences = (latent_posterior_dists.log_prob(latent_posterior_samples)
                                - latent_prior_dists.log_prob(latent_posterior_samples))
    latent_kl_divergences = tf.reduce_sum(latent_kl_divergences, axis=1)
    outputs.update({
      'latent_kl_divergence': tf.reduce_mean(latent_kl_divergences),
    })

    elbo = - latent_kl_divergences

    # Compute the reconstruction part of the ELBO
    likelihood_dists = {}
    likelihood_log_probs = {}
    reconstruction_error = {}
    for name in self.reconstruct_names:
      likelihood_dists[name] = self.decoders[name](latent_posterior_samples)
      images_tmp = tf.image.convert_image_dtype(images[name], tf.float32)
      likelihood_log_probs[name] = likelihood_dists[name].log_prob(images_tmp)
      likelihood_log_probs[name] = tf.reduce_sum(likelihood_log_probs[name], axis=1)
      reconstruction_error[name] = tf.reduce_sum(tf.square(images_tmp - likelihood_dists[name].distribution.loc),
                                         axis=list(range(-len(likelihood_dists[name].event_shape), 0)))
      reconstruction_error[name] = tf.reduce_sum(reconstruction_error[name], axis=1)
      outputs.update({
        'log_likelihood_'+name: tf.reduce_mean(likelihood_log_probs[name]),
        'reconstruction_error_'+name: tf.reduce_mean(reconstruction_error[name]),
      })
      elbo += likelihood_log_probs[name]

    # average over the batch dimension
    loss = -tf.reduce_mean(elbo)

    # Generate the images
    posterior_images = {}
    prior_images = {}
    conditional_prior_images = {}
    for name in self.reconstruct_names:
      posterior_images[name] = likelihood_dists[name].mean()
      prior_images[name] = self.decoders[name](latent_prior_samples).mean()
      conditional_prior_images[name] = self.decoders[name](latent_conditional_prior_samples).mean()

    images = tf.concat([tf.image.convert_image_dtype(images[k], tf.float32)
      for k in list(set(self.input_names+self.reconstruct_names))], axis=-2)
    posterior_images = tf.concat(list(posterior_images.values()), axis=-2)
    prior_images = tf.concat(list(prior_images.values()), axis=-2)
    conditional_prior_images = tf.concat(list(conditional_prior_images.values()), axis=-2)

    outputs.update({
      'elbo': tf.reduce_mean(elbo),
      'images': images,
      'posterior_images': posterior_images,
      'prior_images': prior_images,
      'conditional_prior_images': conditional_prior_images,
    })
    return loss, outputs

  def sample_prior_or_posterior(self, actions, step_types=None, images=None):
    """Samples from the prior latents, or latents conditioned on input images."""
    if step_types is None:
      batch_size = tf.shape(actions)[0]
      sequence_length = actions.shape[1]  # should be statically defined
      step_types = tf.fill(
          [batch_size, sequence_length + 1], ts.StepType.MID)
    else:
      sequence_length = step_types.shape[1] - 1
      actions = actions[:, :sequence_length]
    if images is not None:
      features = self.get_features(images)

    # Swap batch and time axes
    actions = tf.transpose(actions, [1, 0, 2])
    step_types = tf.transpose(step_types, [1, 0])
    if images is not None:
      features = tf.transpose(features, [1, 0, 2])

    # Get latent distributions and samples
    latent_dists = []
    latent_samples = []
    for t in range(sequence_length + 1):
      is_conditional = images is not None and (t < list(images.values())[0].shape[1])
      if t == 0:
        if is_conditional:
          latent_dist = self.latent_first_posterior(features[t])
        else:
          latent_dist = self.latent_first_prior(step_types[t])  # step_types is only used to infer batch_size
        latent_sample = latent_dist.sample()
      else:
        reset_mask = tf.equal(step_types[t], ts.StepType.FIRST)
        reset_mask = tf.expand_dims(reset_mask, 1)
        if is_conditional:
          latent_first_dist = self.latent_first_posterior(features[t])
          latent_dist = self.latent_posterior(features[t], latent_samples[t-1], actions[t-1])
        else:
          latent_first_dist = self.latent_first_prior(step_types[t])
          latent_dist = self.latent_prior(latent_samples[t-1], actions[t-1])
        latent_dist = nest_utils.map_distribution_structure(
            functools.partial(tf.where, reset_mask), latent_first_dist, latent_dist)
        latent_sample = latent_dist.sample()

      latent_dists.append(latent_dist)
      latent_samples.append(latent_sample)

    latent_dists = nest_utils.map_distribution_structure(lambda *x: tf.stack(x, axis=1), *latent_dists)
    latent_samples = tf.stack(latent_samples, axis=1)
    return latent_samples, latent_dists

  def sample_posterior(self, images, actions, step_types, features=None):
    """Sample posterior latents conditioned on input images."""
    sequence_length = step_types.shape[1] - 1
    actions = actions[:, :sequence_length]

    if features is None:
      features = self.get_features(images)

    # Swap batch and time axes
    features = tf.transpose(features, [1, 0, 2])
    actions = tf.transpose(actions, [1, 0, 2])
    step_types = tf.transpose(step_types, [1, 0])

    # Get latent distributions and samples
    latent_dists = []
    latent_samples = []
    for t in range(sequence_length + 1):
      if t == 0:
        latent_dist = self.latent_first_posterior(features[t])
        latent_sample = latent_dist.sample()
      else:
        reset_mask = tf.equal(step_types[t], ts.StepType.FIRST)
        reset_mask = tf.expand_dims(reset_mask, 1)
        latent_first_dist = self.latent_first_posterior(features[t])
        latent_dist = self.latent_posterior(features[t], latent_samples[t-1], actions[t-1])
        latent_dist = nest_utils.map_distribution_structure(
            functools.partial(tf.where, reset_mask), latent_first_dist, latent_dist)
        latent_sample = latent_dist.sample()

      latent_dists.append(latent_dist)
      latent_samples.append(latent_sample)

    latent_dists = nest_utils.map_distribution_structure(lambda *x: tf.stack(x, axis=1), *latent_dists)
    latent_samples = tf.stack(latent_samples, axis=1)
    return latent_samples, latent_dists
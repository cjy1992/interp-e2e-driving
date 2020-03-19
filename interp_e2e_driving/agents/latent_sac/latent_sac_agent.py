# Copyright (c) 2020: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gin
import tensorflow as tf

from tf_agents.agents import tf_agent
from tf_agents.policies import greedy_policy
from tf_agents.trajectories import trajectory
from tf_agents.utils import eager_utils

from interp_e2e_driving.policies import latent_actor_policy
from interp_e2e_driving.utils import gif_utils
from interp_e2e_driving.utils import nest_utils

LatentSacLossInfo = collections.namedtuple(
    'LatentSacLossInfo', ('sac_loss', 'model_loss'))


@gin.configurable
class LatentSACAgent(tf_agent.TFAgent):
  """TF agent with a sequential latent model and inner SAC agent."""
  
  def __init__(self,
               time_step_spec,
               action_spec,
               inner_agent,
               model_network,
               model_optimizer,
               model_batch_size=None,
               num_images_per_summary=1,
               sequence_length=2,
               gradient_clipping=None,
               summarize_grads_and_vars=False,
               train_step_counter=None,
               fps=10,
               name=None):
    tf.Module.__init__(self, name=name)

    self._inner_agent = inner_agent
    self._model_network = model_network
    self._model_optimizer = model_optimizer
    self._model_batch_size = model_batch_size
    self._num_images_per_summary = num_images_per_summary
    self._gradient_clipping = gradient_clipping
    self._summarize_grads_and_vars = summarize_grads_and_vars
    self._train_step_counter = train_step_counter
    self._fps = fps

    policy = latent_actor_policy.LatentActorPolicy(
      time_step_spec=time_step_spec,
      action_spec=action_spec,
      inner_policy=inner_agent.collect_policy,
      model_network=model_network,
      collect=False)

    super(LatentSACAgent, self).__init__(
        time_step_spec,
        action_spec,
        policy=policy,
        collect_policy=policy,
        train_sequence_length=sequence_length + 1,
        train_step_counter=train_step_counter,
        )

  def _experience_to_transitions(self, experience):
    transitions = trajectory.to_transition(experience)
    time_steps, policy_steps, next_time_steps = transitions
    actions = policy_steps.action
    return time_steps, actions, next_time_steps

  def _train(self, experience, weights=None):
    """Train both the inner sac agent with the sequential latent model."""
    
    # Get the sequence with shape [B,T,...]
    time_steps, actions, next_time_steps = self._experience_to_transitions(
        experience)
    # Get the last transition (s,a,s') with shape [B,1,...]
    time_step, action, next_time_step = self._experience_to_transitions(
        tf.nest.map_structure(lambda x: x[:, -2:], experience))
    # Squeeze to shape [B,...]
    time_step, action, next_time_step = tf.nest.map_structure(
        lambda x: tf.squeeze(x, axis=1), (time_step, action, next_time_step))

    # If persistent=False, can only require gradient for one time.
    # If watch_accessed_variables=False, must indicate which variables to request gradient
    with tf.GradientTape() as tape:
      # Sample the latent from model network
      images = experience.observation
      latent_samples_and_dists = self._model_network.sample_posterior(
          images, actions, experience.step_type)
      latents, _ = latent_samples_and_dists
      if isinstance(latents, (tuple, list)):
        latents = tf.concat(latents, axis=-1)
      # Shape [B,...]
      latent, next_latent = tf.unstack(latents[:, -2:], axis=1)

      model_loss = self.model_loss(
          images,
          experience.action,
          experience.step_type,
          latent_posterior_samples_and_dists=latent_samples_and_dists,
          weights=weights)

    tf.debugging.check_numerics(model_loss, 'Model loss is inf or nan.')
    trainable_model_variables = self._model_network.trainable_variables
    assert trainable_model_variables, ('No trainable model variables to '
                                          'optimize.')
    model_grads = tape.gradient(model_loss, trainable_model_variables)
    self._apply_gradients(model_grads, trainable_model_variables, self._model_optimizer)

    latent_experience = trajectory.Trajectory(
      step_type=experience.step_type,
      observation=tf.stop_gradient(latents),
      action=experience.action,
      policy_info=experience.policy_info,
      next_step_type=experience.next_step_type,
      reward=experience.reward,
      discount=experience.discount)

    # latent_experience = latent_experience[:, -2:]
    latent_experience = tf.nest.map_structure(lambda x: x[:, -2:], latent_experience)

    sac_loss = self._inner_agent.train(latent_experience, weights).loss
    total_loss = sac_loss + model_loss

    extra = LatentSacLossInfo(sac_loss=sac_loss,
                              model_loss=model_loss)

    return tf_agent.LossInfo(loss=total_loss, extra=extra)


  def _apply_gradients(self, gradients, variables, optimizer):
    grads_and_vars = list(zip(gradients, variables))
    if self._gradient_clipping is not None:
      grads_and_vars = eager_utils.clip_gradient_norms(grads_and_vars,
                                                       self._gradient_clipping)

    if self._summarize_grads_and_vars:
      eager_utils.add_variables_summaries(grads_and_vars,
                                          self.train_step_counter)
      eager_utils.add_gradients_summaries(grads_and_vars,
                                          self.train_step_counter)

    optimizer.apply_gradients(grads_and_vars)

  def model_loss(self,
                 images,
                 actions,
                 step_types,
                 latent_posterior_samples_and_dists=None,
                 weights=None):
      with tf.name_scope('model_loss'):
        if self._model_batch_size is not None:
          actions, step_types = tf.nest.map_structure(
              lambda x: x[:self._model_batch_size],
              (actions, step_types))
          images_new = {}
          for k, v in images.items():
            images_new[k] = v[:self._model_batch_size]
          if latent_posterior_samples_and_dists is not None:
            latent_posterior_samples, latent_posterior_dists = latent_posterior_samples_and_dists
            latent_posterior_samples = tf.nest.map_structure(
                lambda x: x[:self._model_batch_size], latent_posterior_samples)
            latent_posterior_dists = nest_utils.map_distribution_structure(
                lambda x: x[:self._model_batch_size], latent_posterior_dists)
            latent_posterior_samples_and_dists = (
                latent_posterior_samples, latent_posterior_dists)

        model_loss, outputs = self._model_network.compute_loss(
            images_new, actions, step_types,
            latent_posterior_samples_and_dists=latent_posterior_samples_and_dists)
        for name, output in outputs.items():
          if output.shape.ndims == 0:
            tf.summary.scalar(name, output, step=self.train_step_counter)
          elif output.shape.ndims == 5:
            output = output[:self._num_images_per_summary]
            output = tf.transpose(output, [1,0,2,3,4])
            output = tf.reshape(output, [output.shape[0], output.shape[1]*output.shape[2], output.shape[3], output.shape[4]])
            output = tf.expand_dims(output, axis=0)
            gif_utils.gif_summary(name, output, self._fps,
                         saturate=True, step=self.train_step_counter)
          else:
            raise NotImplementedError

        if weights is not None:
          model_loss *= weights

        model_loss = tf.reduce_mean(input_tensor=model_loss)

        return model_loss
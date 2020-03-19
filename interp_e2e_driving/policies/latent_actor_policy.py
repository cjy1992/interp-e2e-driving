# Copyright (c) 2020: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents.ppo import ppo_utils
from tf_agents.networks import network
from tf_agents.policies import tf_policy
from tf_agents.specs import distribution_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step


@gin.configurable
class LatentActorPolicy(tf_policy.Base):
  """Class to build Actor Policies."""

  def __init__(self,
               time_step_spec,
               action_spec,
               inner_policy,
               model_network,
               collect=True,
               clip=True,
               name=None):
    """Builds an Actor Policy given a actor network.
    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      inner_policy: A tf policy that takes the latent state as input.
      model_network: The sequential latent model to infer latent states.
      collect: Whether this is a collect policy.
      clip: Whether to clip actions to spec before returning them.  Default
        True. Most policy-based algorithms (PCL, PPO, REINFORCE) use unclipped
        continuous actions for training.
      name: The name of this policy. All variables in this module will fall
        under that name. Defaults to the class name.
    Raises:
      ValueError: if actor_network is not of type network.Network.
    """
    self._inner_policy = inner_policy
    self._model_network = model_network
    self._collect = collect

    # Info spec
    info_spec = ()
    if collect:
      if isinstance(inner_policy._actor_network, network.DistributionNetwork):
        network_output_spec = inner_policy._actor_network.output_spec
      else:
        network_output_spec = tf.nest.map_structure(
            distribution_spec.deterministic_distribution_from_spec, action_spec)
      info_spec = tf.nest.map_structure(lambda spec: spec.input_params_spec,
                                        network_output_spec)

    # Get latent spec
    latent_spec = tensor_spec.TensorSpec((model_network.latent_size,),
                                          dtype=tf.float32)
    # The policy will now store the state for actor network, and the latent state
    policy_state_spec = (inner_policy._actor_network.state_spec, latent_spec, action_spec)

    super(LatentActorPolicy, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        policy_state_spec=policy_state_spec,
        info_spec=info_spec,
        clip=clip,
        name=name)

  def _get_initial_state(self, batch_size):
    """The initial state, which is the prior."""
    network_state = tensor_spec.zero_spec_nest(
        self._policy_state_spec[0],
        outer_dims=None if batch_size is None else [batch_size])
    latent_state = self._model_network.sample_prior(batch_size)
    last_action = tensor_spec.zero_spec_nest(
        self._policy_state_spec[2],
        outer_dims=None if batch_size is None else [batch_size])
    return (network_state, latent_state, last_action)

  def _apply_actor_network(self, time_step, policy_state):
    """Generate action using actor network with latent."""
    network_state, latent_state, _ = policy_state
    return self._inner_policy._actor_network(
        latent_state, time_step.step_type, network_state)

  def _variables(self):
    variables = self._inner_policy.variables()
    variables += list(self._model_network)
    return variables

  def _action(self, time_step, policy_state, seed):
    """This will update the state based on time_step and generate action."""
    distribution_step = self._distribution(time_step, policy_state)
    action = distribution_step.action.sample(seed=seed)

    # Update the last action to policy state
    network_state, latent_state, _ = distribution_step.state
    policy_state = (network_state, latent_state, action)

    return distribution_step._replace(action=action, state=policy_state)

  def _distribution(self, time_step, policy_state):
    network_state, latent_state, last_action = policy_state
    latent_state = tf.where(time_step.is_first(), 
      self._model_network.first_filter(time_step.observation),
      self._model_network.filter(time_step.observation, latent_state, last_action))

    # Update the latent state
    policy_state = (network_state, latent_state, last_action)
    # Actor network outputs nested structure of distributions or actions.
    actions_or_distributions, network_state = self._apply_actor_network(
        time_step, policy_state)
    # Update the network state
    policy_state = (network_state, latent_state, last_action)

    def _to_distribution(action_or_distribution):
      if isinstance(action_or_distribution, tf.Tensor):
        # This is an action tensor, so wrap it in a deterministic distribution.
        return tfp.distributions.Deterministic(loc=action_or_distribution)
      return action_or_distribution

    distributions = tf.nest.map_structure(_to_distribution,
                                          actions_or_distributions)

    # Prepare policy_info.
    if self._collect:
      policy_info = ppo_utils.get_distribution_params(distributions)
    else:
      policy_info = ()

    return policy_step.PolicyStep(distributions, policy_state, policy_info)
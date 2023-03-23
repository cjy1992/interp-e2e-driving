# This file is modified from <https://github.com/tensorflow/agents>
from absl import logging

import gin
import tensorflow as tf
from tf_agents.keras_layers import dynamic_unroll_layer
from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step
from tf_agents.utils import nest_utils

from tensorflow.python.util import nest

def _copy_layer(layer):
  """Create a copy of a Keras layer with identical parameters.
  The new layer will not share weights with the old one.
  Args:
    layer: An instance of `tf.keras.layers.Layer`.
  Returns:
    A new keras layer.
  Raises:
    TypeError: If `layer` is not a keras layer.
    ValueError: If `layer` cannot be correctly cloned.
  """
  if not isinstance(layer, tf.keras.layers.Layer):
    raise TypeError('layer is not a keras layer: %s' % str(layer))

  # pylint:disable=unidiomatic-typecheck
  if type(layer) == tf.compat.v1.keras.layers.DenseFeatures:
    raise ValueError('DenseFeatures V1 is not supported. '
                     'Use tf.compat.v2.keras.layers.DenseFeatures instead.')
  if layer.built:
    logging.warn(
        'Beware: Copying a layer that has already been built: \'%s\'.  '
        'This can lead to subtle bugs because the original layer\'s weights '
        'will not be used in the copy.', layer.name)
  # Get a fresh copy so we don't modify an incoming layer in place.  Weights
  # will not be shared.
  return type(layer).from_config(layer.get_config())


@gin.configurable
class MultiInputsCriticRnnNetwork(network.Network):
  """Creates a recurrent critic network with multiple source inputs."""

  def __init__(self,
               input_tensor_spec,
               preprocessing_layers=None,
               preprocessing_combiner=None,
               action_fc_layer_params=(200,),
               joint_fc_layer_params=(100,),
               lstm_size=(40,),
               output_fc_layer_params=(200, 100),
               activation_fn=tf.keras.activations.relu,
               name='MultiInputsCriticRnnNetwork'):
    """Creates an instance of `MultiInputsCriticRnnNetwork`.
    Args:
      input_tensor_spec: A tuple of (observation, action) each of type
        `tensor_spec.TensorSpec` representing the inputs.
      preprocessing_layers: (Optional.) A nest of `tf.keras.layers.Layer`
        representing preprocessing for the different observations.
        All of these layers must not be already built. For more details see
        the documentation of `networks.EncodingNetwork`.
      preprocessing_combiner: (Optional.) A keras layer that takes a flat list
        of tensors and combines them. Good options include
        `tf.keras.layers.Add` and `tf.keras.layers.Concatenate(axis=-1)`.
        This layer must not be already built. For more details see
        the documentation of `networks.EncodingNetwork`.
      action_fc_layer_params: Optional list of parameters for a fully_connected
        layer to apply to the actions, where each item is the number of units
        in the layer.
      joint_fc_layer_params: Optional list of parameters for a fully_connected
        layer to apply after merging observations and actions, where each item
        is the number of units in the layer.
      lstm_size: An iterable of ints specifying the LSTM cell sizes to use.
      output_fc_layer_params: Optional list of fully_connected parameters, where
        each item is the number of units in the layer. This is applied after the
        LSTM cell.
      activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
      name: A string representing name of the network.
    Returns:
      A tf.float32 Tensor of q-values.
    Raises:
      ValueError: If `observation_spec` or `action_spec` contains more than one
        item.
    """
    observation_spec, action_spec = input_tensor_spec

    if len(tf.nest.flatten(action_spec)) > 1:
      raise ValueError('Only a single action is supported by this network.')

    if preprocessing_layers is None:
      flat_preprocessing_layers = None
    else:
      flat_preprocessing_layers = [
          _copy_layer(layer) for layer in tf.nest.flatten(preprocessing_layers)
      ]
      # Assert shallow structure is the same. This verifies preprocessing
      # layers can be applied on expected input nests.
      observation_nest = observation_spec
      # Given the flatten on preprocessing_layers above we need to make sure
      # input_tensor_spec is a sequence for the shallow_structure check below
      # to work.
      if not nest.is_sequence(observation_spec):
        observation_nest = [observation_spec]
      nest.assert_shallow_structure(
          preprocessing_layers, observation_nest, check_types=False)

    if (len(tf.nest.flatten(observation_spec)) > 1 and
        preprocessing_combiner is None):
      raise ValueError(
          'preprocessing_combiner layer is required when more than 1 '
          'observation_spec is provided.')

    if preprocessing_combiner is not None:
      preprocessing_combiner = _copy_layer(preprocessing_combiner)

    action_layers = utils.mlp_layers(
        None,
        action_fc_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1. / 3., mode='fan_in', distribution='uniform'),
        name='action_encoding')

    joint_layers = utils.mlp_layers(
        None,
        joint_fc_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1. / 3., mode='fan_in', distribution='uniform'),
        name='joint_mlp')

    # Create RNN cell
    if len(lstm_size) == 1:
      cell = tf.keras.layers.LSTMCell(lstm_size[0])
    else:
      cell = tf.keras.layers.StackedRNNCells(
          [tf.keras.layers.LSTMCell(size) for size in lstm_size])

    counter = [-1]

    def create_spec(size):
      counter[0] += 1
      return tensor_spec.TensorSpec(
          size, dtype=tf.float32, name='network_state_%d' % counter[0])

    state_spec = tf.nest.map_structure(create_spec, cell.state_size)

    output_layers = utils.mlp_layers(fc_layer_params=output_fc_layer_params,
                                     name='output')

    output_layers.append(
        tf.keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.003, maxval=0.003),
            name='value'))

    super(MultiInputsCriticRnnNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=state_spec,
        name=name)

    self._action_layers = action_layers
    self._joint_layers = joint_layers
    self._dynamic_unroll = dynamic_unroll_layer.DynamicUnroll(cell)
    self._output_layers = output_layers

    self._preprocessing_nest = tf.nest.map_structure(lambda l: None,
                                                     preprocessing_layers)
    self._flat_preprocessing_layers = flat_preprocessing_layers
    self._preprocessing_combiner = preprocessing_combiner

  def call(self, inputs, step_type, network_state=None, training=False):
    observation, action = inputs
    observation_spec, _ = self.input_tensor_spec

    # Preprocess for multiple observations
    if self._flat_preprocessing_layers is None:
      processed = observation
    else:
      processed = []
      for obs, layer in zip(
          nest.flatten_up_to(
              self._preprocessing_nest, observation, check_types=False),
          self._flat_preprocessing_layers):
        processed.append(layer(obs, training=training))
      if len(processed) == 1 and self._preprocessing_combiner is None:
        # If only one observation is passed and the preprocessing_combiner
        # is unspecified, use the preprocessed version of this observation.
        processed = processed[0]
    observation = processed
    if self._preprocessing_combiner is not None:
      observation = self._preprocessing_combiner(observation)
    observation_spec = tensor_spec.TensorSpec((observation.shape[-1],), dtype=observation.dtype)

    num_outer_dims = nest_utils.get_outer_rank(observation,
                                               observation_spec)
    if num_outer_dims not in (1, 2):
      raise ValueError(
          'Input observation must have a batch or batch x time outer shape.')

    has_time_dim = num_outer_dims == 2
    if not has_time_dim:
      # Add a time dimension to the inputs.
      observation = tf.nest.map_structure(lambda t: tf.expand_dims(t, 1),
                                          observation)
      action = tf.nest.map_structure(lambda t: tf.expand_dims(t, 1), action)
      step_type = tf.nest.map_structure(lambda t: tf.expand_dims(t, 1),
                                        step_type)

    observation = tf.cast(tf.nest.flatten(observation)[0], tf.float32)
    action = tf.cast(tf.nest.flatten(action)[0], tf.float32)

    batch_squash = utils.BatchSquash(2)  # Squash B, and T dims.
    observation = batch_squash.flatten(observation)  # [B, T, ...] -> [BxT, ...]
    action = batch_squash.flatten(action)

    for layer in self._action_layers:
      action = layer(action, training=training)

    joint = tf.concat([observation, action], -1)
    for layer in self._joint_layers:
      joint = layer(joint, training=training)

    joint = batch_squash.unflatten(joint)  # [B x T, ...] -> [B, T, ...]

    with tf.name_scope('reset_mask'):
      reset_mask = tf.equal(step_type, time_step.StepType.FIRST)
    # Unroll over the time sequence.
    joint, network_state = self._dynamic_unroll(
        joint,
        reset_mask,
        initial_state=network_state,
        training=training)

    output = batch_squash.flatten(joint)  # [B, T, ...] -> [B x T, ...]

    for layer in self._output_layers:
      output = layer(output, training=training)

    q_value = tf.reshape(output, [-1])
    q_value = batch_squash.unflatten(q_value)  # [B x T, ...] -> [B, T, ...]
    if not has_time_dim:
      q_value = tf.squeeze(q_value, axis=1)

    return q_value, network_state
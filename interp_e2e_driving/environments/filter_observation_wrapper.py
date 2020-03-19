# Copyright (c) 2020: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import gym
import numpy as np
import tensorflow as tf
from tf_agents.environments import wrappers

class FilterObservationWrapper(wrappers.PyEnvironmentBaseWrapper):
  """Environment wrapper to filter observation channels."""

  def __init__(self, gym_env, input_channels):
    super(FilterObservationWrapper, self).__init__(gym_env)
    self.input_channels = input_channels

    observation_spaces = collections.OrderedDict()
    for channel in self.input_channels:
      observation_spaces[channel] = self._env.observation_space[channel]
    self.observation_space = gym.spaces.Dict(observation_spaces)

  def _modify_observation(self, observation):
    observations = collections.OrderedDict()
    for channel in self.input_channels:
      observations[channel] = observation[channel]
    return observations

  def _step(self, action):
    observation, reward, done, info = self._env.step(action)
    observation = self._modify_observation(observation)
    return observation, reward, done, info

  def _reset(self):
    observation = self._env.reset()
    return self._modify_observation(observation)
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

import inspect

import tensorflow as tf


def map_distribution_structure(func, *dist_structure):
  def _get_params(dist):
    return {k: v for k, v in dist.parameters.items() if
            isinstance(v, tf.Tensor)}

  def _get_other_params(dist):
    return {k: v for k, v in dist.parameters.items() if
            not isinstance(v, tf.Tensor)}

  def _func(*dist_list):
    # all dists should be instances of the same class
    for dist in dist_list[1:]:
      assert dist.__class__ == dist_list[0].__class__
    dist_ctor = dist_list[0].__class__

    dist_other_params_list = [_get_other_params(dist) for dist in dist_list]

    # all dists should have the same non-tensor params
    for dist_other_params in dist_other_params_list[1:]:
      assert dist_other_params == dist_other_params_list[0]
    dist_other_params = dist_other_params_list[0]

    # filter out params that are not in the constructor's signature
    sig = inspect.signature(dist_ctor)
    dist_other_params = {k: v for k, v in dist_other_params.items() if
                         k in sig.parameters}

    dist_params_list = [_get_params(dist) for dist in dist_list]
    values_list = [list(params.values()) for params in dist_params_list]
    values_list = list(zip(*values_list))

    structure_list = [func(*values) for values in values_list]

    values_list = [tf.nest.flatten(structure) for structure in structure_list]
    values_list = list(zip(*values_list))
    dist_params_list = [dict(zip(dist_params_list[0].keys(), values)) for values
                        in values_list]
    dist_list = [dist_ctor(**params, **dist_other_params) for params in
                 dist_params_list]

    dist_structure = tf.nest.pack_sequence_as(structure_list[0], dist_list)
    return dist_structure

  return tf.nest.map_structure(_func, *dist_structure)
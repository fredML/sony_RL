import jax
import jax.numpy as jnp
import haiku as hk
import chex
from typing import *
from math import prod

class ValueNetwork(hk.Module):
  def __init__(self, name: Optional[str] = None) -> None:
    super().__init__(name=name)

  def __call__(self, s: chex.Array, a: chex.Array) -> chex.Array:
    '''h = s
    h = hk.Linear(400)(h)
    h = jax.nn.relu(h)
    h = hk.Linear(300)(jnp.concatenate((h,a), axis=1))
    h = jax.nn.relu(h)
    return hk.Linear(1, hk.initializers.RandomUniform(-3e-3, 3e-3))(h)[..., 0]'''
    h = hk.nets.MLP([32, 32, 32])(jnp.concatenate((s,a), axis=1))
    return hk.Linear(1, hk.initializers.RandomUniform(-3e-3, 3e-3))(h)[...,0]

class PolicyNetwork(hk.Module):
  def __init__(self, action_shape, name: Optional[str] = None) -> None:
    super().__init__(name=name)
    self.action_shape = action_shape

  def __call__(self, s: chex.Array) -> chex.Array:
    action_dims = prod(self.action_shape)
    h = hk.nets.MLP([32, 32, 32])(s)
    h = hk.Linear(action_dims, hk.initializers.RandomUniform(-3e-3, 3e-3))(h)
    h = jnp.tanh(h)
    return h
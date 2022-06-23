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
    return hk.nets.MLP([32,32,32,1])(jnp.concatenate((s,a), axis=1))[...,0]


class PolicyNetwork(hk.Module):
  def __init__(self, action_shape, name: Optional[str] = None) -> None:
    super().__init__(name=name)
    self.action_shape = action_shape

  def __call__(self, s: chex.Array) -> chex.Array:
    action_dims = prod(self.action_shape)
    h = hk.nets.MLP([32,32,32, action_dims])(s)
    h = jax.nn.tanh(h)
    return h
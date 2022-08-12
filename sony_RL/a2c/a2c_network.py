import jax
import jax.numpy as jnp
import haiku as hk
import chex
from typing import *
from math import prod

class ValueNetwork(hk.Module):
  def __init__(self, name: Optional[str] = None) -> None:
    super().__init__(name=name)

  def __call__(self, s: chex.Array) -> chex.Array:
    return hk.nets.MLP([32,32,32,1])(s)[...,0]

class PolicyNetwork(hk.Module):
  def __init__(self, action_shape, name: Optional[str] = None) -> None:
    super().__init__(name=name)
    self.action_shape = action_shape

  def __call__(self, s: chex.Array ) -> Tuple[chex.Array, chex.Array]:
    action_dims = prod(self.action_shape)
    h = hk.nets.MLP([32,32,32, 2 * action_dims])(s)
    h = jax.nn.tanh(h)
    mu, sigma = jnp.split(h, 2, axis=-1)
    sigma = jax.nn.softplus(sigma)
    return 0.1*mu, 0.1*sigma

'''class ValueNetwork(hk.Module):
  def __init__(self, name: Optional[str] = None) -> None:
    super().__init__(name=name)

  def __call__(self, s: chex.Array) -> chex.Array:
    model = hk.Sequential([
        hk.Conv2D(32, kernel_shape=[3,3], stride=1, padding='SAME'),
        jax.nn.relu,
        hk.MaxPool(2, strides=1, padding='SAME'),
        hk.Conv2D(64, kernel_shape=[2,2], stride=1, padding='SAME'),
        jax.nn.relu,
        hk.Flatten(),
    ])

    h = model(s)

    return hk.Linear(1)(h)[...,0]

class PolicyNetwork(hk.Module):
  def __init__(self, action_shape, name: Optional[str] = None) -> None:
    super().__init__(name=name)
    self.action_shape = action_shape

  def __call__(self, s: chex.Array ) -> Tuple[chex.Array, chex.Array]:
    action_dims = prod(self.action_shape)
    model = hk.Sequential([
        hk.Conv2D(32, kernel_shape=[3,3], stride=1, padding='SAME'),
        jax.nn.relu,
        hk.MaxPool(2, strides=1, padding='SAME'),
        hk.Conv2D(64, kernel_shape=[2,2], stride=1, padding='SAME'),
        jax.nn.relu,
        hk.Flatten(),
    ])

    h = model(s)
    h = hk.Linear(2 * action_dims)(h)
    h = jax.nn.tanh(h)
    mu, sigma = jnp.split(h, 2, axis=-1)
    sigma = jax.nn.softplus(sigma)
    return mu, sigma'''

'''class ValueNetwork(hk.Module):
  def __init__(self, name: Optional[str] = None) -> None:
    super().__init__(name=name)

  def __call__(self, s: chex.Array) -> chex.Array:

    im, vol = s

    model_im = hk.Sequential([
        hk.Conv2D(8, kernel_shape=[3,3], stride=1, padding='SAME'),
        jax.nn.relu,
        hk.MaxPool(2, strides=1, padding='SAME'),
        hk.Conv2D(16, kernel_shape=[2,2], stride=1, padding='SAME'),
        jax.nn.relu,
        hk.MaxPool(2, strides=1, padding='SAME'),
        hk.Flatten(),
        hk.nets.MLP([64])
    ])

    model_vol = hk.Sequential([
        hk.Conv3D(8, kernel_shape=3, stride=1, padding='SAME'),
        jax.nn.relu,
        hk.MaxPool(2, strides=1, padding='SAME'),
        hk.Conv3D(16, kernel_shape=3, stride=1, padding='SAME'),
        jax.nn.relu,
        hk.MaxPool(2, strides=1, padding='SAME'),
        hk.Flatten(),
        hk.nets.MLP([64])
    ])

    h1 = model_vol(vol)
    h2 = model_im(im)
    h = jnp.concatenate((h1,h2),axis=-1)

    return hk.Linear(1)(h)[...,0]
    
class PolicyNetwork(hk.Module):
  def __init__(self, action_shape, name: Optional[str] = None) -> None:
    super().__init__(name=name)
    self.action_shape = action_shape

  def __call__(self, s: chex.Array ) -> Tuple[chex.Array, chex.Array]:
    action_dims = prod(self.action_shape)
    im, vol = s

    model_im = hk.Sequential([
        hk.Conv2D(8, kernel_shape=[3,3], stride=1, padding='SAME'),
        jax.nn.relu,
        hk.MaxPool(2, strides=1, padding='SAME'),
        hk.Conv2D(16, kernel_shape=[2,2], stride=1, padding='SAME'),
        jax.nn.relu,
        hk.MaxPool(2, strides=1, padding='SAME'),
        hk.Flatten(),
        hk.nets.MLP([64])
    ])

    model_vol = hk.Sequential([
        hk.Conv3D(8, kernel_shape=3, stride=1, padding='SAME'),
        jax.nn.relu,
        hk.MaxPool(2, strides=1, padding='SAME'),
        hk.Conv3D(16, kernel_shape=3, stride=1, padding='SAME'),
        jax.nn.relu,
        hk.MaxPool(2, strides=1, padding='SAME'),
        hk.Flatten(),
        hk.nets.MLP([64])
    ])

    h1 = model_vol(vol)
    h2 = model_im(im)
    h = jnp.concatenate((h1,h2),axis=-1)

    h = hk.Linear(2 * action_dims)(h)
    h = jax.nn.tanh(h)
    mu, sigma = jnp.split(h, 2, axis=-1)
    sigma = jax.nn.softplus(sigma)
    return mu, sigma'''

import haiku as hk
import jax 
from typing import *
import jax.numpy as jnp

def dqn_state(s, num_actions, is_training):
    h = hk.Conv2D(32, kernel_shape=[3,3], stride=1, padding='SAME')(s)
    #h = hk.MaxPool(2, strides=1, padding='SAME')(h)
    h = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate = 0.98)(h, is_training)
    h = jax.nn.relu(h)

    h = hk.Conv2D(32, kernel_shape=[3,3], stride=1, padding='SAME')(h)
    h = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate = 0.98)(h, is_training)
    h = jax.nn.relu(h)

    h = hk.Conv2D(32, kernel_shape=[3,3], stride=1, padding='VALID')(h)
    h = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate = 0.98)(h, is_training)

    h = hk.Flatten()(h)
    h = hk.nets.MLP([32, 32, num_actions])(h)

    return h

def dqn(s,num_actions):
    model = hk.Sequential([
        hk.Conv2D(8, kernel_shape=3, stride=1, padding='SAME'),
        jax.nn.relu,
        hk.MaxPool(2, strides=1, padding='SAME'),
        hk.Conv2D(16, kernel_shape=3, stride=1, padding='SAME'),
        jax.nn.relu,
        hk.Flatten(),
        hk.nets.MLP([32,32,num_actions])
    ])

    return model(s)

def dqn_low(s,num_actions):
    return hk.nets.MLP([32,32,num_actions])(s)

def dqn_3d(s,num_actions):
    pos, vol = s

    model_vol = hk.Sequential([
        hk.Conv3D(8, kernel_shape=3, stride=1, padding='SAME'),
        jax.nn.relu,
        hk.MaxPool(2, strides=1, padding='SAME'),
        hk.Conv3D(8, kernel_shape=3, stride=1, padding='SAME'),
        jax.nn.relu,
        hk.MaxPool(2, strides=1, padding='SAME'),
        hk.Flatten(),
        hk.nets.MLP([32,32])
    ])

    h1 = model_vol(vol)
    h1 = jnp.tanh(h1)
    h2 = hk.nets.MLP([32,32,32])(pos)
    h2 = jnp.tanh(h2)
    h = jnp.concatenate((h1,h2),axis=-1)
    return hk.nets.MLP([num_actions])(h)

def dueling_catch_network(s,num_actions):

    model = hk.Sequential([
            hk.Conv2D(32, kernel_shape=[3,3], stride=1, padding='SAME'),
            jax.nn.relu,
            hk.MaxPool(2, strides=1, padding='SAME'),
            hk.Conv2D(64, kernel_shape=[2,2], stride=1, padding='SAME'),
            jax.nn.relu,
            hk.MaxPool(2, strides=1, padding='SAME'),
            hk.Flatten()
        ])

    torso_out = model(s)

    value = hk.nets.MLP([32, 32, 1])(torso_out)
    advantage = hk.nets.MLP([32, 32, num_actions])(torso_out)

    q_values =  value + (advantage - jnp.mean(advantage, axis=-1, keepdims=True))

    return q_values

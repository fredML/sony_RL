import math
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
import numpy as np
from jax import nn


@jax.jit
def gaussian_log_prob(
    log_std: jnp.ndarray,
    noise: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculate log probabilities of gaussian distributions.
    """
    return -0.5 * (jnp.square(noise) + 2 * log_std + jnp.log(2 * math.pi))


@jax.jit
def gaussian_and_tanh_log_prob(
    log_std: jnp.ndarray,
    noise: jnp.ndarray,
    action: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculate log probabilities of gaussian distributions and tanh transformation.
    """
    return gaussian_log_prob(log_std, noise) - jnp.log(nn.relu(1.0 - jnp.square(action)) + 1e-6)


@jax.jit
def evaluate_gaussian_and_tanh_log_prob(
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
    action: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculate log probabilities of gaussian distributions and tanh transformation given samples.
    """
    noise = (jnp.arctanh(action) - mean) / (jnp.exp(log_std) + 1e-8)
    return gaussian_and_tanh_log_prob(log_std, noise, action).sum(axis=1, keepdims=True)


@partial(jax.jit, static_argnums=3)
def reparameterize_gaussian(
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
    key: jnp.ndarray,
    return_log_pi: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample from gaussian distributions.
    """
    std = jnp.exp(log_std)
    noise = jax.random.normal(key, std.shape)
    action = mean + noise * std
    if return_log_pi:
        return action, gaussian_log_prob(log_std, noise).sum(axis=1, keepdims=True)
    else:
        return action


@partial(jax.jit, static_argnums=3)
def reparameterize_gaussian_and_tanh(
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
    key: jnp.ndarray,
    return_log_pi: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample from gaussian distributions and tanh transforamation.
    """
    std = jnp.exp(log_std)
    noise = jax.random.normal(key, std.shape)
    action = mean + noise * std
    if return_log_pi:
        #return action, gaussian_and_tanh_log_prob(log_std, noise, action).sum(axis=1, keepdims=True)
        log_pi = norm.logpdf(action.flatten(), loc=mean.flatten(), scale=std.flatten()+1e-6) #add small eps to avoid nan values
        #log_pi = jnp.maximum(log_pi, -1e3) 
        return jnp.tanh(action), log_pi.reshape(std.shape).sum(axis=1, keepdims=True) - jnp.log(1-jnp.tanh(action)**2).sum(axis=1, keepdims=True)
    else:
        return jnp.tanh(action)


@jax.jit
def calculate_kl_divergence(
    p_mean: np.ndarray,
    p_std: np.ndarray,
    q_mean: np.ndarray,
    q_std: np.ndarray,
) -> jnp.ndarray:
    """
    Calculate KL Divergence between gaussian distributions.
    """
    var_ratio = jnp.square(p_std / q_std)
    t1 = jnp.square((p_mean - q_mean) / q_std)
    return 0.5 * (var_ratio + t1 - 1 - jnp.log(var_ratio))

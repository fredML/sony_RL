from a2c_network import PolicyNetwork, ValueNetwork

import os
path = os.getcwd()
os.chdir('/mnt/diskSustainability/frederic/sony_RL/base_functions')

from data import LearnerState, Transition
os.chdir(path)

import chex
from acme import types
from typing import *
import numpy as np
import jax
import jax.numpy as jnp
import optax
import haiku as hk
import abc
import rlax
import functools

LogsDict = Mapping[str, chex.Array]


# A very simple agent API, with just enough to interact with the environment
# and to update its potential parameters.
class Agent(abc.ABC):
    @abc.abstractmethod
    def learner_step(self, transition: Transition) -> Mapping[str, chex.ArrayNumpy]:
        """One step of learning on a trajectory.

        The mapping returned can contain various logs.
        """
        pass


class A2CAgent(Agent):
    def __init__(self, seed: int, learning_rate: float, gamma: float, entropy_loss_coef: float, env_shape: tuple, action_shape: tuple) -> None:
        self._rng = jax.random.PRNGKey(seed=seed)
        self._init_loss, apply_loss = hk.without_apply_rng(
            hk.transform(self._loss_function))
        self._grad = jax.grad(apply_loss, has_aux=True)
        _, self._apply_policy = hk.without_apply_rng(
            hk.transform(self._hk_apply_policy))
        _, self._apply_value = hk.without_apply_rng(
            hk.transform(self._hk_apply_value))

        self._entropy_loss_coef = entropy_loss_coef
        self._gamma = gamma
        self._action_shape = action_shape
        self._env_shape = env_shape

        self._optimizer = optax.adam(learning_rate=learning_rate)
        self.init_fn = jax.jit(self._init_fn)
        self.update_fn = jax.jit(self._update_fn)
        self.apply_policy = jax.jit(self._apply_policy)
        self.apply_value = jax.jit(self._apply_value)

        self._rng, init_rng = jax.random.split(self._rng)
        self._learner_state = self._init_fn(
            init_rng, self._generate_dummy_transition())

    def _init_fn(self, rng: chex.PRNGKey, transition: Transition) -> LearnerState:
        params = self._init_loss(rng, transition)
        opt_state = self._optimizer.init(params)
        return LearnerState(params=params, opt_state=opt_state)

    def _update_fn(self, learner_state: LearnerState, transition: Transition) -> Tuple[LearnerState, LogsDict]:
        grads, aux = self._grad(learner_state.params, transition)
        updates, new_opt_state = self._optimizer.update(
            grads, learner_state.opt_state)
        new_params = optax.apply_updates(learner_state.params, updates)
        return LearnerState(params=new_params, opt_state=new_opt_state), aux

    def learner_step(self, transition: Transition) -> Mapping[str, chex.ArrayNumpy]:
        self._learner_state, logs = self.update_fn(
            self._learner_state, transition)
        return logs
    
    def _actor_step(self, params, rng: chex.PRNGKey, observations: types.NestedArray, for_eval: bool = False) -> types.NestedArray:
        mu, sigma = self.apply_policy(params, observations)
        
        if for_eval:
            actions = rlax.gaussian_diagonal().sample(rng, mu, 0. * sigma) #no stochastic effect when eval mode
        else:
            actions = rlax.gaussian_diagonal().sample(rng, mu, sigma) #variable actions pi(s) for one given observation
        return actions

    def actor_step(self, observations, for_eval = False):
        actor_params = {k: self._learner_state.params[k] for k in self._learner_state.params.keys(
        ) if k.startswith("policy")}
        return self._actor_step(actor_params, self._rng, observations, for_eval)

    def _generate_dummy_transition(self) -> Transition:

        observation = jnp.array(np.random.uniform(-1, 1, self._env_shape), dtype=jnp.float32)

        action = jnp.array(np.random.uniform(-1, 1, self._action_shape), dtype=jnp.float32)

        return Transition(
            obs_tm1=observation[None],
            action_tm1=action,
            reward_t=jnp.zeros(1),
            discount_t=jnp.zeros(1),
            obs_t=observation[None],
            done=jnp.zeros(1))


    def _hk_apply_value(self, observations: types.NestedArray):
        return ValueNetwork()(observations)

    def _hk_apply_policy(self, observations: types.NestedArray):
        return PolicyNetwork(self._action_shape)(observations)

    def _loss_function(self, transition: Transition) -> Tuple[chex.Array, LogsDict]:
        # inputs are assumed to be provided such that the full sequence that we get is
        # o_0 a_0 r_0 d_0, ...., o_T, a_T, r_T, d_T
        mu, sigma = PolicyNetwork(self._action_shape)(transition.obs_tm1)

        obs = jnp.concatenate((transition.obs_tm1, transition.obs_t[-1:]),axis=0)

        '''values_tm1 = ValueNetwork()(transition.obs_tm1)
        values = ValueNetwork()(transition.obs_t)
        values = jax.lax.stop_gradient(values)'''

        values = ValueNetwork()(obs)

        batched_return_fn = jax.vmap(
          functools.partial(rlax.lambda_returns, stop_target_gradients=True),
          in_axes=1,
          out_axes=1)
        value_targets = batched_return_fn(
          transition.reward_t[...,None],
          (self._gamma * transition.discount_t[...,None] * (1. - transition.done[...,None])),
          values[1:],
          )

        '''value_targets = transition.reward_t + \
            (self._gamma * transition.discount_t * (1. - transition.done))*values[1:]'''
        value_loss = jnp.mean(.5 * jnp.square(values[:-1] - value_targets))

        sg_advantages = jax.lax.stop_gradient(value_targets - values[:-1])
        action_log_probs = rlax.gaussian_diagonal().logprob(transition.action_tm1, mu, sigma)
        #action_probs = PolicyNetwork(self._num_actions)(transition.obs_tm1)
        #action_log_probs = jnp.log(batched_indexing(
        #    action_probs, jnp.int16(transition.action_tm1)))
        entropies = rlax.gaussian_diagonal().entropy(mu, sigma)
        entropy_loss = -jnp.mean(entropies)
        policy_loss = -jnp.mean(sg_advantages * action_log_probs)
        policy_loss = policy_loss + self._entropy_loss_coef * entropy_loss

        logs = dict(actions_mean=transition.action_tm1.mean(axis=0),
                    value_loss=value_loss,
                    policy_loss=policy_loss,
                    entropy_loss=entropy_loss,
                    value_mean=values.mean(),
                    value_target_mean=value_targets.mean(),
                    mean_reward=transition.reward_t.mean(),
                    mu = mu.mean(),
                    sigma = sigma.mean()
                    )
        return value_loss + policy_loss, logs

from ddpg_network import PolicyNetwork, ValueNetwork

import os
path = os.getcwd()
os.chdir('/mnt/diskSustainability/frederic/sony_RL/base_functions')

from data import Transition
os.chdir(path)

import chex
from acme import types
from typing import *
import jax
import jax.numpy as jnp
import optax
import haiku as hk
import abc
from acme import specs

LogsDict = Mapping[str, chex.Array]

@chex.dataclass
class LearnerState:
    params: chex.Array
    opt_actor_state: optax.OptState
    opt_critic_state: optax.OptState

# A very simple agent API, with just enough to interact with the environment
# and to update its potential parameters.
class Agent(abc.ABC):
    @abc.abstractmethod
    def learner_step(self, transition: Transition) -> Mapping[str, chex.ArrayNumpy]:
        """One step of learning on a trajectory.

        The mapping returned can contain various logs.
        """
        pass

class DDPGAgent(Agent):
    '''
    Implement a Deep Deterministic Policy Gradient agent, as described in the paper 'Continious control with deep reinforcement learning'
    
    '''
    def __init__(self, seed: int, actor_learning_rate: float, critic_learning_rate: float, gamma: float, tau: float, environment_spec: specs.EnvironmentSpec) -> None:
        """
        :param seed: random seed
        :param actor_learning_rate: lr of actor
        :param critic_learning_rate: lr of actor
        :param gamma: discount factor
        :param tau: slow copy factor (1: full copy at each step, 0: copy never happens)
        :param environment_spec: environment info
        """
        self._rng = jax.random.PRNGKey(seed=seed)
        self._init_critic_loss, apply_critic_loss = hk.without_apply_rng(hk.transform(self._critic_loss_function))
        self._init_actor_loss, apply_actor_loss = hk.without_apply_rng(hk.transform(self._actor_loss_function))       
       
        self._grad_actor = jax.grad(apply_critic_loss, has_aux=True)
        self._grad_critic = jax.grad(apply_actor_loss, has_aux=True)

        _, self._apply_policy = hk.without_apply_rng(hk.transform(self._hk_apply_policy))
        _, self._apply_value = hk.without_apply_rng(hk.transform(self._hk_apply_value))

        #self._entropy_loss_coef = entropy_loss_coef  don't need that as we are not regulazing
        self._gamma = gamma
        self._tau = tau
        self._environment_spec = environment_spec
        self._action_shape = self._environment_spec.actions.shape
        self._env_shape = self._environment_spec.observations.shape

        self._critic_optimizer = optax.adam(learning_rate=critic_learning_rate)
        self._actor_optimizer = optax.adam(learning_rate=actor_learning_rate)

        self.init_fn = jax.jit(self._init_fn)
        self.update_fn = jax.jit(self._update_fn)
        self.apply_policy = jax.jit(self._apply_policy)
        self.apply_value = jax.jit(self._apply_value)

        self._rng, init_rng = jax.random.split(self._rng)
        self._learner_state = self.init_fn(init_rng, self._generate_dummy_transition())

    def _generate_dummy_transition(self) -> Transition:

        observation = jax.random.uniform(key=self._rng, minval=-1, maxval=1, shape=self._env_shape, dtype=jnp.float32)
        action = jax.random.uniform(key=self._rng, minval=-1, maxval=1, shape=self._action_shape, dtype=jnp.float32)

        return Transition(
            obs_tm1=observation[None],
            action_tm1=action[None],
            reward_t=jnp.zeros(1),
            discount_t=jnp.zeros(1),
            obs_t=observation[None],
            done=jnp.zeros(1))

    def _init_fn(self, rng: chex.PRNGKey, transition: Transition) -> LearnerState:
        """
        Initializes the networks and the optimizers, such that the target and main networks are equal.
        """
        critic_params = self._init_critic_loss(rng, transition)
        actor_params = self._init_actor_loss(rng, transition)

        # get only critic, resp. actor params. 
        critic_params = {k:critic_params[k] for k in critic_params.keys() if  k.startswith("value/")}
        actor_params = {k:actor_params[k] for k in actor_params.keys() if  k.startswith("policy/")}
        # copy the params for the target networks. 
        target_critic_params = {k.replace("value/", "value_target/"):critic_params[k] for k in critic_params.keys()}
        target_actor_params = {k.replace("policy/", "policy_target/"):actor_params[k] for k in actor_params.keys()}

        opt_critic_state = self._critic_optimizer.init(critic_params)
        opt_actor_state = self._actor_optimizer.init(actor_params)

        actor_params.update(critic_params)
        actor_params.update(target_actor_params)
        actor_params.update(target_critic_params)

        return LearnerState(params=actor_params, opt_critic_state=opt_critic_state, opt_actor_state=opt_actor_state)

    def slow_copy(self, params):
        """
        Performs slow copy between main and target networks parameters (or state)
        :param params: networks parameters (or state)
        :return: updated networks parameters (or state)
        """
        tau = self._tau
        for k, v in params.items():
            if k.startswith(("value/", "policy/")):
                k_t = k.replace("value/", "value_target/").replace("policy/", "policy_target/")
                params[k_t] = jax.tree_util.tree_map(lambda x, y: tau * x + (1 - tau) * y, v, params[k_t])
        return params

    def _update_fn(self, learner_state: LearnerState, transition: Transition) -> Tuple[LearnerState, LogsDict]:
        """
        Performs a learning step.
        :param learner_state: parameters and state of networks and optimizer
        :param transition: input transition
        :return: updated parameters and state of networks and optimizer, logs
        """
        # critic update
        #   Compute the critic loss and the gradients for ALL the parameters value, policy and targets
        critic_grads, aux_critic = self._grad_critic(learner_state.params, transition)

        # Select the only params and gradients we are interested in for this part, the critics
        critic_params = {k:learner_state.params[k] for k in learner_state.params.keys() if  k.startswith("value/")}
        critic_grads_only_value = {k:critic_grads[k] for k in critic_grads.keys() if k.startswith("value/")}

        # Perform one optimization step
        critic_updates, new_opt_critic_state = self._critic_optimizer.update(critic_grads_only_value, learner_state.opt_critic_state, critic_params)
        new_critic_params = optax.apply_updates(critic_params, critic_updates)

        # Update the critic
        learner_state.params.update(new_critic_params)

        # actor update
        #   Compute the actor loss and the gradients for ALL the parameters value, policy and targets
        actor_grads, aux_actor = self._grad_actor(learner_state.params, transition)

        # Select the only params and gradients we are interested in for this part, the actors
        actor_params = {k:learner_state.params[k] for k in learner_state.params.keys() if  k.startswith("policy/")}
        actor_grads_only_policy = {k:actor_grads[k] for k in actor_grads.keys() if k.startswith("policy/")}

        # Perform one optimization step
        actor_updates, new_opt_actor_state = self._actor_optimizer.update(actor_grads_only_policy, learner_state.opt_actor_state, actor_params)
        new_actor_params = optax.apply_updates(actor_params, actor_updates)

        # Update the critic
        learner_state.params.update(new_actor_params)

        # Update the target networks using slow copy
        new_params = self.slow_copy(learner_state.params) 

        # aggregate the logs
        aux_critic.update(aux_actor)

        return LearnerState(params=new_params, opt_critic_state=new_opt_critic_state, opt_actor_state=new_opt_actor_state), aux_critic

    def learner_step(self, transition: Transition) -> Mapping[str, chex.ArrayNumpy]:
        self._learner_state, logs = self.update_fn(self._learner_state, transition)
        return logs

    def _actor_step(self, learner_state: LearnerState, observations: types.NestedArray,
                            ) -> types.NestedArray:
        actions = self.apply_policy(learner_state.params, observations)
        return actions #learner state does not need to be propagated as it's an off-policy algorithm

    def actor_step(self, observations: types.NestedArray) -> types.NestedArray:
        """Returns actions in response to observations."""
        return self._actor_step(self._learner_state, observations)

    def _hk_apply_value(self, observations: types.NestedArray, actions:types.NestedArray) -> chex.Array:
        return ValueNetwork(name='value')(observations, actions) #2nd arg will be actions

    def _hk_apply_policy(self, observations: types.NestedArray) -> chex.Array:
        return PolicyNetwork(self._action_shape, name='policy')(observations)

    def _actor_loss_function(self, transition: Transition) -> Tuple[chex.Array, LogsDict]:
        # actor loss
        action_ = PolicyNetwork(self._action_shape, name='policy')(transition.obs_tm1)
        values = ValueNetwork(name='value')(transition.obs_tm1, action_) #2nd arg will be actions
        actor_loss = -jnp.mean(values)

        logs = dict(policy_loss=actor_loss)
        return actor_loss, logs


    def _critic_loss_function(self, transition: Transition) -> Tuple[chex.Array, LogsDict]:
        #critic loss
        q_value = ValueNetwork(name='value')(transition.obs_tm1, transition.action_tm1)
        next_action = PolicyNetwork(self._action_shape, name='policy_target')(transition.obs_t)
        bootstrapped_q = ValueNetwork(name='value_target')(transition.obs_t, next_action)
        q_target = jax.lax.stop_gradient(transition.reward_t + self._gamma *  (1 - transition.done) * bootstrapped_q)
       
        value_loss = jnp.mean(.5 * jnp.square(q_target - q_value))

        logs = dict(actions_mean=transition.action_tm1.mean(axis=0),
                    value_loss=value_loss,
                    value_mean=q_value.mean(),
                    value_target_mean=q_target.mean(),
                    mean_reward=transition.reward_t.mean(),
                    )
        return value_loss, logs
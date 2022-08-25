from functools import partial
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from actor_critic import OffPolicyActorCritic
from critic import ContinuousQFunction
from actor import DeterministicPolicy
from optim import optimize
from preprocess import add_noise

print(jax.devices())

class DDPG(OffPolicyActorCritic):
    name = "DDPG"

    def __init__(
        self,
        num_agent_steps,
        state_space,
        action_space,
        seed,
        encoder=None,
        max_grad_norm=None,
        gamma=0.99,
        nstep=1,
        num_critics=1,
        buffer_size=10 ** 3,
        use_per=False,
        batch_size=32,
        start_steps=10000,
        update_interval=1,
        tau=5e-3,
        fn_actor=None,
        fn_critic=None,
        lr_actor=1e-5,
        lr_critic=1e-4,
        units_actor=(256, 256),
        units_critic=(256, 256),
        d2rl=False,
        std=0.1,
        update_interval_policy=1000,
    ):
        super(DDPG, self).__init__(
            num_agent_steps=num_agent_steps,
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            max_grad_norm=max_grad_norm,
            gamma=gamma,
            nstep=nstep,
            num_critics=num_critics,
            buffer_size=buffer_size,
            use_per=use_per,
            batch_size=batch_size,
            start_steps=start_steps,
            update_interval=update_interval,
            tau=tau,
        )
        if d2rl:
            self.name += "-D2RL"

        if fn_critic is None:

            def fn_critic(s, a):
                return ContinuousQFunction(
                    num_critics=num_critics,
                    hidden_units=units_critic,
                    d2rl=d2rl,
                )(s, a)

        if fn_actor is None:

            def fn_actor(s):
                return DeterministicPolicy(
                    action_space=action_space,
                    hidden_units=units_actor,
                    d2rl=d2rl,
                )(s)
        
        critic_init, critic_apply = hk.without_apply_rng(hk.transform(fn_critic))
        self.critic_apply_jit = jax.jit(critic_apply)
        actor_init, actor_apply = hk.without_apply_rng(hk.transform(fn_actor))
        self.actor_apply_jit = jax.jit(actor_apply)

        dummy_state = np.random.uniform(0,1,state_space.shape)[None]
        dummy_action = np.random.uniform(-1,1,len(action_space.shape))[None]

        self.encoder = encoder
        if encoder is not None:
            vae_apply_jit, params_vae, bn_vae_state = self.encoder
            dummy_state, _ = vae_apply_jit(params_vae, bn_vae_state, np.random.uniform(0,1,state_space.shape), False)
            dummy_state = dummy_state[2]
            dummy_state = dummy_state.reshape((1,-1))

        # Critic.
        self.params_critic = self.params_critic_target = critic_init(next(self.rng), 
                                                                          dummy_state,
                                                                          dummy_action)                                                                                
        self.params_actor = self.params_actor_target = actor_init(next(self.rng), dummy_state)

        opt_init, self.opt_critic = optax.radam(lr_critic)
        self.opt_state_critic = opt_init(self.params_critic)
        
        opt_init, self.opt_actor = optax.radam(lr_actor)
        self.opt_state_actor = opt_init(self.params_actor)

        # Other parameters.
        self.std = std
        self.update_interval_policy = update_interval_policy

    @partial(jax.jit, static_argnums=0)
    def _select_action(
        self,
        params_actor: hk.Params,
        state: np.ndarray,
    ) -> jnp.ndarray:
        if self.encoder is not None:
            state = jnp.reshape(state, (-1, *self.state_space.shape[1:]))
            vae_apply_jit, params_vae, bn_vae_state = self.encoder
            state, _ = vae_apply_jit(params_vae, bn_vae_state, state, False)
            state = state[2]
            state = jnp.reshape(state, (1, -1))
        return self.actor_apply_jit(params_actor, state)

    @partial(jax.jit, static_argnums=0)
    def _explore(
        self,
        params_actor: hk.Params,
        state: np.ndarray,
        key: jnp.ndarray,
    ) -> jnp.ndarray:
        if self.encoder is not None:
            state = jnp.reshape(state, (-1, *self.state_space.shape[1:]))
            vae_apply_jit, params_vae, bn_vae_state = self.encoder
            state, _ = vae_apply_jit(params_vae, bn_vae_state, state, False)
            state = state[2]
            state = jnp.reshape(state, (1, -1))
        action = self.actor_apply_jit(params_actor, state)
        return add_noise(action, key, self.std, -jnp.pi/2, jnp.pi/2)

    def update(self, writer=None):
        self.learning_step += 1
        weight, batch = self.buffer.sample(self.batch_size)
        state, action, reward, done, next_state = batch

        if self.encoder is not None:

            state = jnp.reshape(state, (-1, *self.state_space.shape[1:])) #need to reshape from (bs, k, img_size, img_size, 1) to (bs*k,...)
            next_state = jnp.reshape(next_state, (-1, *self.state_space.shape[1:]))
            vae_apply_jit, params_vae, bn_vae_state = self.encoder

            state, _ = vae_apply_jit(params_vae, bn_vae_state, state, False)
            state = state[2]
            next_state, _ = vae_apply_jit(params_vae, bn_vae_state, next_state, False)
            next_state = next_state[2]

            state = jnp.reshape(state, (self.batch_size, -1)) # output of vae is (bs*k, latent_dim), need to reshape (bs,k*latent_dim)
            next_state = jnp.reshape(next_state, (self.batch_size, -1))

        # Update critic and target.
        self.opt_state_critic, self.params_critic, loss_critic, (abs_td, target, q_val) = optimize(
            self._loss_critic,
            self.opt_critic,
            self.opt_state_critic,
            self.params_critic,
            self.max_grad_norm,
            params_actor_target=self.params_actor_target,
            params_critic_target=self.params_critic_target,
            state=state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state,
            weight=weight,
            **self.kwargs_critic,
        )
        self.params_critic_target = self._update_target(self.params_critic_target, self.params_critic)

        # Update priority.
        if self.use_per:
            self.buffer.update_priority(abs_td)

        # Update actor and target.
        if self.agent_step % self.update_interval_policy == 0:
            self.opt_state_actor, self.params_actor, loss_actor, _ = optimize(
                self._loss_actor,
                self.opt_actor,
                self.opt_state_actor,
                self.params_actor,
                self.max_grad_norm,
                params_critic=self.params_critic,
                state=state,
                **self.kwargs_actor,
            )
            self.params_actor_target = self._update_target(self.params_actor_target, self.params_actor)

            if writer and self.agent_step % 1000 == 0:
                writer.add_scalar("loss/critic", loss_critic, self.agent_step)
                writer.add_scalar("loss/actor", loss_actor, self.agent_step)
                writer.add_scalar("episode/target_q", target.mean(), self.agent_step)
                writer.add_scalar("episode/mean_q", q_val.mean(), self.agent_step)
                writer.add_scalar("episode/done", done.mean(), self.agent_step)
                writer.add_scalar("episode/reward", reward.mean(), self.agent_step)

    @partial(jax.jit, static_argnums=0)
    def _sample_action(
        self,
        params_actor: hk.Params,
        state: np.ndarray,
    ) -> jnp.ndarray:
        if (self.encoder is not None) & (len(state.shape) > 2):
            state = jnp.reshape(state, (-1, *self.state_space.shape[1:]))
            vae_apply_jit, params_vae, bn_vae_state = self.encoder
            state, _ = vae_apply_jit(params_vae, bn_vae_state, state, False)
            state = state[2]
            state = jnp.reshape(state, (1, -1))
        return self.actor_apply_jit(params_actor, state)

    @partial(jax.jit, static_argnums=0)
    def _calculate_target(
        self,
        params_critic_target: hk.Params,
        reward: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
        next_action: jnp.ndarray,
    ) -> jnp.ndarray:
        next_q = self._calculate_value(params_critic_target, next_state, next_action)
        return jax.lax.stop_gradient(reward + (1.0 - done) * self.discount * next_q)

    @partial(jax.jit, static_argnums=0)
    def _loss_critic(
        self,
        params_critic: hk.Params,
        params_critic_target: hk.Params,
        params_actor_target: hk.Params,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
        weight: np.ndarray,
        *args,
        **kwargs,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        next_action = self._sample_action(params_actor_target, next_state, *args, **kwargs)
        target = self._calculate_target(params_critic_target, reward, done, next_state, next_action)
        q_list = self._calculate_value_list(params_critic, state, action)
        q_val = q_list[0]
        loss_critic, abs_td = self._calculate_loss_critic_and_abs_td(q_list, target, weight)
        return loss_critic, (abs_td, target, q_val)

    @partial(jax.jit, static_argnums=0)
    def _loss_actor(
        self,
        params_actor: hk.Params,
        params_critic: hk.Params,
        state: np.ndarray,
    ) -> jnp.ndarray:
        action = self.actor_apply_jit(params_actor, state)
        mean_q = self.critic_apply_jit(params_critic, state, action)[0].mean()
        return -mean_q, None
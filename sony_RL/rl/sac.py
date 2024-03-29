from functools import partial
from typing import List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from actor_critic import OffPolicyActorCritic
from actor import StateDependentGaussianPolicy
from critic import ContinuousQFunction
from optim import optimize
from distribution import reparameterize_gaussian_and_tanh, gaussian_and_tanh_log_prob

print(jax.devices())


class SAC(OffPolicyActorCritic):
    name = "SAC"

    def __init__(
        self,
        num_agent_steps,
        state_space,
        action_space,
        seed,
        encoder=None,
        use_goal=False,
        max_grad_norm=None,
        scale_reward=1,
        gamma=0.99,
        nstep=1,
        num_critics=2,
        buffer_size=10 ** 3,
        use_per=False,
        batch_size=32,
        start_steps=1000,
        update_interval=1,
        update_interval_target=None,
        tau=5e-3,
        fn_actor=None,
        fn_critic=None,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        units_actor=(256, 256),
        units_critic=(256, 256),
        log_std_min=-20.0,
        log_std_max=1.0,
        d2rl=False,
        init_alpha=1,
        adam_b1_alpha=0.9,
        *args,
        **kwargs,
    ):
        if not hasattr(self, "use_key_critic"):
            self.use_key_critic = True
        if not hasattr(self, "use_key_actor"):
            self.use_key_actor = True

        super(SAC, self).__init__(
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
            *args,
            **kwargs,
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
                return StateDependentGaussianPolicy(
                    action_space=action_space,
                    hidden_units=units_actor,
                    log_std_min=log_std_min,
                    log_std_max=log_std_max,
                    d2rl=d2rl,
                )(s)

        self.use_goal = use_goal

        critic_init, critic_apply = hk.without_apply_rng(hk.transform(fn_critic))
        self.critic_apply_jit = jax.jit(critic_apply)
        actor_init, actor_apply = hk.without_apply_rng(hk.transform(fn_actor))
        self.actor_apply_jit = jax.jit(actor_apply)

        dummy_state = np.random.uniform(0,1,state_space.shape)[None]
        dummy_action = np.random.uniform(-1,1,len(action_space.shape))[None]

        self.encoder = encoder
        if encoder is not None:
            vae_apply_jit, params_vae, bn_vae_state = self.encoder
            if self.use_goal:
                dummy_state, dummy_goal = dummy_state[:,:-1,...], dummy_state[:,-1,0,0]
            dummy_state = dummy_state[0]
            dummy_state, _ = vae_apply_jit(params_vae, bn_vae_state, dummy_state, False)
            dummy_state = dummy_state['latent_eval']
            if len(dummy_state.shape)>2:
                dummy_state = jnp.argmax(dummy_state, axis=-1)
            dummy_state = jnp.tanh(dummy_state)
            dummy_state = dummy_state.reshape((1,-1))
            if self.use_goal:
                dummy_state = jnp.concatenate((dummy_state, dummy_goal), axis=1)
        
        self.params_critic = self.params_critic_target = critic_init(next(self.rng), 
                                                                          dummy_state,
                                                                          dummy_action)                                                                                
        self.params_actor = actor_init(next(self.rng), dummy_state)
    
        opt_init, self.opt_critic = optax.radam(lr_critic)
        self.opt_state_critic = opt_init(self.params_critic)
        
        opt_init, self.opt_actor = optax.radam(lr_actor)
        self.opt_state_actor = opt_init(self.params_actor)

        self.target_entropy = -np.prod(self.action_space.shape)
        self.log_alpha = jnp.array(np.log(init_alpha), dtype=jnp.float32)
        opt_init, self.opt_alpha = optax.radam(lr_alpha, b1=adam_b1_alpha)
        self.opt_state_alpha = opt_init(self.log_alpha)

        self.scale_reward = scale_reward
        self.update_interval_target = update_interval_target

    @partial(jax.jit, static_argnums=0)
    def _select_action(
        self,
        params_actor: hk.Params,
        state: np.ndarray,
    ) -> jnp.ndarray:
        if self.use_goal:
            state, goal = state[:,:-1,...], state[:,-1,0,0]
        if self.encoder is not None:
            state = jnp.reshape(state, (-1, *self.state_space.shape[1:]))
            vae_apply_jit, params_vae, bn_vae_state = self.encoder
            state, _ = vae_apply_jit(params_vae, bn_vae_state, state, False)
            state = state['latent_eval']
            if len(state.shape)>2:
                state = jnp.argmax(state, axis=-1)
            state = jnp.tanh(state)
            state = jnp.reshape(state, (1, -1))
            if self.use_goal:
                state = jnp.concatenate((state, goal), axis=1)
        mean, _ = self.actor_apply_jit(params_actor, state)
        return jnp.tanh(mean)

    @partial(jax.jit, static_argnums=0)
    def _explore(
        self,
        params_actor: hk.Params,
        state: np.ndarray,
        key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if self.use_goal:
            state, goal = state[:,:-1,...], state[:,-1,0,0]
        if self.encoder is not None:
            state = jnp.reshape(state, (-1, *self.state_space.shape[1:]))
            vae_apply_jit, params_vae, bn_vae_state = self.encoder
            state, _ = vae_apply_jit(params_vae, bn_vae_state, state, False)
            state = state['latent_eval']
            if len(state.shape)>2:
                state = jnp.argmax(state, axis=-1)
            state = jnp.tanh(state)
            state = jnp.reshape(state, (1, -1))
            if self.use_goal:
                state = jnp.concatenate((state, goal), axis=1)
        mean, log_std = self.actor_apply_jit(params_actor, state)
        return reparameterize_gaussian_and_tanh(mean, log_std, key, False)

    def update(self, writer=None):
        self.learning_step += 1
        weight, batch = self.buffer.sample(self.batch_size)
        state, action, reward, done, next_state = batch
        reward = reward*self.scale_reward

        if self.use_goal:
            state, goal = state[:,:-1,...], state[:,-1,0,0]
            next_state, next_goal = next_state[:,:-1,...], next_state[:,-1,0,0]

        if self.encoder is not None:

            state = jnp.reshape(state, (-1, *self.state_space.shape[1:])) #need to reshape from (bs, k, img_size, img_size, 1) to (bs*k,...)
            next_state = jnp.reshape(next_state, (-1, *self.state_space.shape[1:]))
            vae_apply_jit, params_vae, bn_vae_state = self.encoder

            state, _ = vae_apply_jit(params_vae, bn_vae_state, state, False)
            state = state['latent_eval']
            next_state, _ = vae_apply_jit(params_vae, bn_vae_state, next_state, False)
            next_state = next_state['latent_eval']

            if len(state.shape)>2:
                state = jnp.argmax(state, axis=-1)
                next_state = jnp.argmax(next_state, axis=-1)

            state = jnp.tanh(state)
            next_state = jnp.tanh(next_state)
            state = jnp.reshape(state, (self.batch_size, -1)) # output of vae is (bs*k, latent_dim), need to reshape (bs,k*latent_dim)
            next_state = jnp.reshape(next_state, (self.batch_size, -1))

            if self.use_goal:
                state = jnp.concatenate((state, goal), axis=1)
                next_state = jnp.concatenate((next_state, next_goal), axis=1)
            
        # Update critic.
        self.opt_state_critic, self.params_critic, loss_critic, (abs_td, target, q_val) = optimize(
            self._loss_critic,
            self.opt_critic,
            self.opt_state_critic,
            self.params_critic,
            self.max_grad_norm,
            params_critic_target=self.params_critic_target,
            params_actor=self.params_actor,
            log_alpha=self.log_alpha,
            state=state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state,
            weight=weight,
            **self.kwargs_critic,
        )

        # Update priority.
        if self.use_per:
            self.buffer.update_priority(abs_td)

        # Update actor.
        self.opt_state_actor, self.params_actor, loss_actor, (mean_log_pi, mean_action, std_action) = optimize(
            self._loss_actor,
            self.opt_actor,
            self.opt_state_actor,
            self.params_actor,
            self.max_grad_norm,
            params_critic=self.params_critic,
            log_alpha=self.log_alpha,
            state=state,
            **self.kwargs_actor,
        )
        # Update alpha.
        self.opt_state_alpha, self.log_alpha, loss_alpha, _ = optimize(
            self._loss_alpha,
            self.opt_alpha,
            self.opt_state_alpha,
            self.log_alpha,
            None,
            mean_log_pi=mean_log_pi,
        )

        # Update target network.
        
        self.params_critic_target = self._update_target(self.params_critic_target, self.params_critic)

        if writer and self.agent_step % 1000 == 0:
            writer.add_scalar("episode/target_q", target.mean(), self.agent_step)
            writer.add_scalar("episode/mean_q", q_val.mean(), self.agent_step)
            writer.add_scalar("episode/done", done.mean(), self.agent_step)
            writer.add_scalar("episode/reward", reward.mean(), self.agent_step)
            writer.add_scalar("episode/mean_action", mean_action.mean(), self.agent_step)
            writer.add_scalar("episode/std_action", std_action.mean(), self.agent_step)
            writer.add_scalar("loss/critic", loss_critic, self.agent_step)
            writer.add_scalar("loss/actor", -loss_actor, self.agent_step)
            writer.add_scalar("loss/alpha", loss_alpha, self.agent_step)
            writer.add_scalar("stat/alpha", jnp.exp(self.log_alpha), self.agent_step)
            writer.add_scalar("stat/entropy", -mean_log_pi, self.agent_step)

    @partial(jax.jit, static_argnums=0)
    def _sample_action(
        self,
        params_actor: hk.Params,
        state: np.ndarray,
        key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        '''if (self.encoder is not None) & (len(state.shape) > 2):
            state = jnp.reshape(state, (-1, *self.state_space.shape[1:]))
            vae_apply_jit, params_vae, bn_vae_state = self.encoder
            state = vae_apply_jit(params_vae, bn_vae_state, state, False)
            state = state['latent_eval']
            state = jnp.reshape(state, (1, -1))'''
        mean, log_std = self.actor_apply_jit(params_actor, state)
        action, log_pi = reparameterize_gaussian_and_tanh(mean, log_std, key, True)
        return action, log_pi, mean, log_std

    @partial(jax.jit, static_argnums=0)
    def _calculate_log_pi(
        self,
        action: jnp.ndarray,
        log_std: jnp.ndarray,
        key: jnp.ndarray
    ) -> jnp.ndarray:
        noise = jax.random.normal(key, log_std.shape)
        return gaussian_and_tanh_log_prob(log_std, noise, action).sum(axis=1, keepdims=True)

    @partial(jax.jit, static_argnums=0)
    def _calculate_target(
        self,
        params_critic_target: hk.Params,
        log_alpha: jnp.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
        next_action: jnp.ndarray,
        next_log_pi: jnp.ndarray,
    ) -> jnp.ndarray:
        next_q = self._calculate_value(params_critic_target, next_state, next_action)
        next_q -= jnp.exp(log_alpha) * next_log_pi
        return jax.lax.stop_gradient(reward + (1.0 - done) * self.discount * next_q)

    @partial(jax.jit, static_argnums=0)
    def _loss_critic(
        self,
        params_critic: hk.Params,
        params_critic_target: hk.Params,
        params_actor: hk.Params,
        log_alpha: jnp.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
        weight: np.ndarray or List[jnp.ndarray],
        *args,
        **kwargs,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        next_action, next_log_pi, _, _ = self._sample_action(params_actor, next_state, *args, **kwargs)
        #action_pi, _, _, _ = self._sample_action(params_actor, state, *args, **kwargs)
        target = self._calculate_target(params_critic_target, log_alpha, reward, done, next_state, next_action, next_log_pi)
        q_list = self._calculate_value_list(params_critic, state, action)
        q_val = jnp.asarray(q_list).min(axis=0)
        loss_critic, abs_td = self._calculate_loss_critic_and_abs_td(q_list, target, weight)
        return loss_critic, (abs_td, target, q_val)

    @partial(jax.jit, static_argnums=0)
    def _loss_actor(
        self,
        params_actor: hk.Params,
        params_critic: hk.Params,
        log_alpha: jnp.ndarray,
        state: np.ndarray,
        *args,
        **kwargs,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        action_pi, log_pi, mean, log_std = self._sample_action(params_actor, state, *args, **kwargs)
        mean_q = self._calculate_value(params_critic, state, action_pi).mean()
        mean_log_pi = log_pi.mean()
        return jax.lax.stop_gradient(jnp.exp(log_alpha)) * mean_log_pi - mean_q, (jax.lax.stop_gradient(mean_log_pi), mean, jnp.exp(log_std))

    @partial(jax.jit, static_argnums=0)
    def _loss_alpha(
        self,
        log_alpha: jnp.ndarray,
        mean_log_pi: jnp.ndarray,
    ) -> jnp.ndarray:
        return -log_alpha * (self.target_entropy + mean_log_pi), None
import os
from functools import partial
from typing import List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from vae import kl_gaussian
from sac import SAC
from critic import ContinuousQFunction
from actor import StateDependentGaussianPolicy
from saving import load_params, save_params
from optim import optimize, soft_update


class SAC_AE(SAC):
    name = "SAC+AE"

    def __init__(
        self,
        num_agent_steps,
        state_space,
        action_space,
        seed,
        encoder,
        max_grad_norm=None,
        gamma=0.99,
        nstep=1,
        num_critics=2,
        buffer_size=10 ** 3,
        use_per=False,
        batch_size=32,
        start_steps=10**3,
        update_interval=1,
        tau=0.001,
        tau_ae=0.01,
        fn_actor=None,
        fn_critic=None,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_ae=1e-3,
        lr_alpha=3e-4,
        units_actor=(32, 32, 32),
        units_critic=(32, 32, 32),
        log_std_min=-10.0,
        log_std_max=2.0,
        d2rl=False,
        init_alpha=0.1,
        adam_b1_alpha=0.9,
        beta=1e-8,
        update_interval_actor=2,
        update_interval_ae=1e2,
        scale_reward=1
    ):
        '''assert len(state_space.shape) == 3 and state_space.shape[:2] == (84, 84)
        assert (state_space.maximum == 255).all()'''
        if d2rl:
            self.name += "-D2RL"

        if fn_critic is None:

            def fn_critic(x, a):
                # Define without linear layer 
                return ContinuousQFunction(
                    num_critics=num_critics,
                    hidden_units=units_critic,
                    d2rl=d2rl,
                )(x, a)

        if fn_actor is None:

            def fn_actor(x):
                # Define without linear layer
                return StateDependentGaussianPolicy(
                    action_space=action_space,
                    hidden_units=units_actor,
                    log_std_min=log_std_min,
                    log_std_max=log_std_max,
                    clip_log_std=False,
                    d2rl=d2rl,
                )(x)

        super().__init__(
            num_agent_steps=num_agent_steps,
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            encoder=encoder,
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
            fn_actor=fn_actor,
            fn_critic=fn_critic,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            lr_alpha=lr_alpha,
            init_alpha=init_alpha,
            adam_b1_alpha=adam_b1_alpha,
        )

        '''filter_sizes = [16,32,64,128]
        n = len(filter_sizes)
        dummy_state = np.random.uniform(0,1,state_space.shape)[None]
        # Encoder.
        def encoder(s, is_training):
            return Encoder(feature_dim, filter_sizes)(s, is_training)

        enc_init, self.enc_apply = hk.without_apply_rng(hk.transform_with_state(encoder))
        self.enc_apply_jit = jax.jit(self.enc_apply, static_argnums=3)
        self.params_encoder, self.bn_enc = enc_init(next(self.rng), dummy_state, True)
        self.params_encoder_target = self.params_encoder

        # Decoder.

        last_conv_shape = state_space.shape[2]//(2**n)
        output_activation = jax.jit(lambda s:s)
        def decoder(s, is_training):
            return Decoder(last_conv_shape, filter_sizes, 1, output_activation)(s, is_training)

        dec_init, dec_apply = hk.without_apply_rng(hk.transform_with_state(decoder))
        self.dec_apply_jit = jax.jit(dec_apply, static_argnums=3)
        self.params_decoder, self.bn_dec = dec_init(next(self.rng), np.random.uniform(-1,1,(1,feature_dim)), True)
        opt_init, self.opt_ae = optax.adam(lr_ae)
        self.opt_state_ae = opt_init(self.params_ae)'''

        self.scale_reward = scale_reward
        self.encoder = encoder
        self.vae_apply_jit, self.params_vae, self.bn_vae_state = self.encoder
        self.params_vae_target = self.params_vae
        opt_init, self.opt_ae = optax.adam(lr_ae)
        self.opt_state_ae = opt_init(self.params_vae)

        # Re-define the optimizer for critic.
        opt_init, self.opt_critic = optax.adam(lr_critic)
        self.opt_state_critic = opt_init(self.params_entire_critic)

        # Other parameters.
        self._update_target_ae = jax.jit(partial(soft_update, tau=tau_ae))
        self.beta = beta
        self.update_interval_actor = update_interval_actor
        self.update_interval_ae = update_interval_ae

    '''def select_action(self, state):
        #last_conv, _ = self._preprocess(self.params_vae, self.bn_vae_state, state[None], False)
        action = self._select_action(self.params_actor, state[None])
        return np.array(action[0])

    def explore(self, state):
        last_conv, _ = self._preprocess(self.params_vae, self.bn_vae_state, state[None], False)
        action = self._explore(self.params_actor, last_conv, next(self.rng))
        return np.array(action[0])'''
    
    @partial(jax.jit, static_argnums=[0,4])
    def _preprocess( 
        self,
        params_vae: hk.Params,
        bn_vae_state,
        state: np.ndarray,
        is_training: bool
    ) -> jnp.ndarray:

        p = len(state)
        state = jnp.reshape(state, (-1, *self.state_space.shape[1:]))
        state, _ = self.vae_apply_jit(params_vae, bn_vae_state, state, is_training)
        state = state[2]
        state = jnp.reshape(state, (p, -1))
        state = jnp.tanh(state)

        return state, bn_vae_state

    def update(self, writer=None):
        self.learning_step += 1
        weight, batch = self.buffer.sample(self.batch_size)
        state, action, reward, done, next_state = batch
        reward = reward*self.scale_reward

        # Update critic.
        self.opt_state_critic, params_entire_critic, loss_critic, (abs_td, target, q_val, self.bn_vae_state) = optimize(
            self._loss_critic,
            self.opt_critic,
            self.opt_state_critic,
            self.params_entire_critic,
            self.max_grad_norm,
            bn_vae_state=self.bn_vae_state,
            params_critic_target=self.params_entire_critic_target,
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
        self.params_vae = params_entire_critic["encoder"]
        self.params_critic = params_entire_critic["critic"]

        # Update priority.
        if self.use_per:
            self.buffer.update_priority(abs_td)

        # Update actor and alpha.
        if self.agent_step % self.update_interval_actor == 0:
            self.opt_state_actor, self.params_actor, loss_actor, (mean_log_pi, mean_action, std_action) = optimize(
                self._loss_actor,
                self.opt_actor,
                self.opt_state_actor,
                self.params_actor,
                self.max_grad_norm,
                bn_vae_state=self.bn_vae_state,
                params_critic=self.params_entire_critic,
                log_alpha=self.log_alpha,
                state=state,
                **self.kwargs_actor,
            )
            self.opt_state_alpha, self.log_alpha, loss_alpha, _ = optimize(
                self._loss_alpha,
                self.opt_alpha,
                self.opt_state_alpha,
                self.log_alpha,
                None,
                mean_log_pi=mean_log_pi,
            )

        # Update autoencoder.
        if self.agent_step % self.update_interval_ae == 0:
            self.opt_state_ae, self.params_vae, loss_vae, (loss_reconst, loss_kl, self.bn_vae_state) = optimize(
                self._loss_ae,
                self.opt_ae,
                self.opt_state_ae,
                self.params_vae,
                self.max_grad_norm,
                bn_vae_state=self.bn_vae_state,
                state=state,
                key=next(self.rng),
            )

        # Update target network.
        self.params_vae_target = self._update_target_ae(self.params_vae_target, self.params_vae)
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
            writer.add_scalar("loss/reconst", loss_reconst, self.agent_step)
            writer.add_scalar("loss/kl", loss_kl, self.agent_step)
            writer.add_scalar("stat/alpha", jnp.exp(self.log_alpha), self.agent_step)
            writer.add_scalar("stat/entropy", -mean_log_pi, self.agent_step)

    @partial(jax.jit, static_argnums=0) # forced to rewrite this base function because we only need subset of params
    def _calculate_value_list(
        self,
        params_critic: hk.Params,
        last_conv: np.ndarray,
        action: np.ndarray,
    ) -> List[jnp.ndarray]:
        return self.critic_apply_jit(params_critic["critic"], last_conv, action)

    @partial(jax.jit, static_argnums=0)
    def _loss_critic(
        self,
        params_critic: hk.Params,
        params_critic_target: hk.Params,
        params_actor: hk.Params,
        bn_vae_state,
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
        feature, bn_vae_state = self._preprocess(params_critic["encoder"], bn_vae_state, state, True)
        next_feature, _ = jax.lax.stop_gradient(self._preprocess(params_critic["encoder"], bn_vae_state, next_state, False))
        res = super(SAC_AE, self)._loss_critic(
            params_critic=params_critic,
            params_critic_target=params_critic_target,
            params_actor=params_actor,
            log_alpha=log_alpha,
            state=feature,
            action=action,
            reward=reward,
            done=done,
            next_state=next_feature,
            weight=weight,
            *args,
            **kwargs,
        )
        loss, aux = res
        aux = (*aux, bn_vae_state)
        return loss, aux

    @partial(jax.jit, static_argnums=0)
    def _loss_actor(
        self,
        params_actor: hk.Params,
        params_critic: hk.Params,
        bn_vae_state,
        log_alpha: jnp.ndarray,
        state: np.ndarray,
        *args,
        **kwargs,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        feature, _ = jax.lax.stop_gradient(self._preprocess(params_critic["encoder"], bn_vae_state, state, False))
        return super(SAC_AE, self)._loss_actor(
            params_actor=params_actor,
            params_critic=params_critic,
            log_alpha=log_alpha,
            state=feature,
            *args,
            **kwargs,
        )

    @partial(jax.jit, static_argnums=0)
    def _loss_ae(
        self,
        params_vae: hk.Params,
        bn_vae_state,
        state: np.ndarray,
        key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Preprocess states.
        #target = preprocess_state(state, key)
        # Reconstruct states.
        state = jnp.reshape(state, (-1, *self.state_space.shape[1:]))
        (recons, latent, mu, log_var), _ = self.vae_apply_jit(params_vae, bn_vae_state, state, True)
        
        # MSE for reconstruction errors.
        loss_reconst = jnp.square(state - recons).mean()
        loss_reconst = loss_reconst/10**3
         
        loss_kl = kl_gaussian(mu, log_var)
        loss_kl = loss_kl * self.beta
        
        return loss_reconst + loss_kl, (loss_reconst, loss_kl, bn_vae_state)

    ''' @property
    def params_ae(self):
        return {
            "encoder": self.params_encoder,
            "decoder": self.params_decoder,
        }'''

    @property
    def params_entire_critic(self):
        return {
            "encoder": self.params_vae,
            "critic": self.params_critic,
        }

    @property
    def params_entire_critic_target(self):
        return {
            "encoder": self.params_vae_target,
            "critic": self.params_critic_target,
        }

    def save_params(self, save_dir):
        super().save_params(save_dir)
        '''save_params(self.params_encoder, os.path.join(save_dir, "params_encoder.npz"))
        save_params(self.params_decoder, os.path.join(save_dir, "params_decoder.npz"))'''
        save_params(self.params_vae, os.path.join(save_dir, "params_vae.npz"))
        save_params(self.bn_vae_state, os.path.join(save_dir, "bn_vae_state.npz"))

    def load_params(self, save_dir):
        super().load_params(save_dir)
        '''self.params_encoder = self.params_encoder_target = load_params(os.path.join(save_dir, "params_encoder.npz"))
        self.params_decoder = load_params(os.path.join(save_dir, "params_decoder.npz"))'''
        self.params_vae = load_params(os.path.join(save_dir, "params_vae.npz"))
        self.bn_vae_state = load_params(os.path.join(save_dir, "bn_vae_state.npz"))
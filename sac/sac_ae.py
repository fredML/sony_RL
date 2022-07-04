import os
from functools import partial
from typing import List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from sac import SAC
from critic import ContinuousQFunction
from actor import StateDependentGaussianPolicy
from misc import SACLinear
from conv import SACDecoder, SACEncoder
from preprocess import preprocess_state
from saving import load_params, save_params
from optim import optimize, soft_update, weight_decay


class SAC_AE(SAC):
    name = "SAC+AE"

    def __init__(
        self,
        num_agent_steps,
        state_space,
        action_space,
        seed,
        max_grad_norm=None,
        gamma=0.99,
        nstep=1,
        num_critics=2,
        buffer_size=10 ** 3,
        use_per=False,
        batch_size=16,
        start_steps=10**3,
        update_interval=1,
        tau=0.01,
        tau_ae=0.05,
        fn_actor=None,
        fn_critic=None,
        lr_actor=1e-4,
        lr_critic=1e-4,
        lr_ae=1e-3,
        lr_alpha=1e-4,
        units_actor=(32, 32),
        units_critic=(32, 32),
        log_std_min=-10.0,
        log_std_max=2.0,
        d2rl=False,
        init_alpha=0.1,
        adam_b1_alpha=0.5,
        feature_dim=5,
        lambda_latent=1e-6,
        lambda_weight=1e-7,
        update_interval_actor=2,
        update_interval_ae=1,
        update_interval_target=2,
    ):
        assert len(state_space.shape) == 3 and state_space.shape[:2] == (84, 84)
        assert (state_space.maximum == 255).all()
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

        fake_feature = jnp.empty((1, feature_dim))
        fake_last_conv = jnp.empty((1, 39200))

        super(SAC_AE, self).__init__(
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
            fn_actor=fn_actor,
            fn_critic=fn_critic,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            lr_alpha=lr_alpha,
            init_alpha=init_alpha,
            adam_b1_alpha=adam_b1_alpha,
        )
        # Encoder.
        self.encoder = hk.without_apply_rng(hk.transform(lambda s: SACEncoder(num_filters=32, num_layers=4)(s)))
        self.params_encoder = self.params_encoder_target = self.encoder.init(next(self.rng), jnp.empty(state_space.shape)[None])

        # Linear layer for critic and decoder.
        self.linear = hk.without_apply_rng(hk.transform(lambda x: SACLinear(feature_dim=feature_dim)(x)))
        self.params_linear = self.params_linear_target = self.linear.init(next(self.rng), fake_last_conv)

        # Decoder.
        self.decoder = hk.without_apply_rng(hk.transform(lambda x: SACDecoder(state_space, num_filters=32, num_layers=4)(x)))
        self.params_decoder = self.decoder.init(next(self.rng), fake_feature)
        opt_init, self.opt_ae = optax.adam(lr_ae)
        self.opt_state_ae = opt_init(self.params_ae)

        # Re-define the optimizer for critic.
        opt_init, self.opt_critic = optax.adam(lr_critic)
        self.opt_state_critic = opt_init(self.params_entire_critic)

        # Other parameters.
        self._update_target_ae = jax.jit(partial(soft_update, tau=tau_ae))
        self.lambda_latent = lambda_latent
        self.lambda_weight = lambda_weight
        self.update_interval_actor = update_interval_actor
        self.update_interval_ae = update_interval_ae
        self.update_interval_target = update_interval_target

    def select_action(self, state):
        last_conv = self._preprocess(self.params_encoder, self.params_linear, state[None, ...])
        action = self._select_action(self.params_actor, last_conv)
        return np.array(action[0])

    def explore(self, state):
        last_conv = self._preprocess(self.params_encoder, self.params_linear, state[None, ...])
        action = self._explore(self.params_actor, last_conv, next(self.rng))
        return np.array(action[0])

    @partial(jax.jit, static_argnums=0)
    def _preprocess(
        self,
        params_encoder: hk.Params,
        params_linear: hk.Params,
        state: np.ndarray,
    ) -> jnp.ndarray:
        conv = self.encoder.apply(params_encoder, state)
        return self.linear.apply(params_linear, conv)

    def update(self, writer=None):
        self.learning_step += 1
        weight, batch = self.buffer.sample(self.batch_size)
        state, action, reward, done, next_state = batch

        # Update critic.
        self.opt_state_critic, params_entire_critic, loss_critic, (abs_td, target, q_val) = optimize(
            self._loss_critic,
            self.opt_critic,
            self.opt_state_critic,
            self.params_entire_critic,
            self.max_grad_norm,
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
        self.params_encoder = params_entire_critic["encoder"]
        self.params_linear = params_entire_critic["linear"]
        self.params_critic = params_entire_critic["critic"]

        # Update priority.
        if self.use_per:
            self.buffer.update_priority(abs_td)

        # Update actor and alpha.
        if self.learning_step % self.update_interval_actor == 0:
            self.opt_state_actor, self.params_actor, loss_actor, mean_log_pi = optimize(
                self._loss_actor,
                self.opt_actor,
                self.opt_state_actor,
                self.params_actor,
                self.max_grad_norm,
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
        if self.learning_step % self.update_interval_ae == 0:
            self.opt_state_ae, params_ae, loss_ae, _ = optimize(
                self._loss_ae,
                self.opt_ae,
                self.opt_state_ae,
                self.params_ae,
                self.max_grad_norm,
                state=state,
                key=next(self.rng),
            )
            self.params_encoder = params_ae["encoder"]
            self.params_linear = params_ae["linear"]
            self.params_decoder = params_ae["decoder"]

        # Update target network.
        if self.learning_step % self.update_interval_target == 0:
            self.params_encoder_target = self._update_target_ae(self.params_encoder_target, self.params_encoder)
            self.params_linear_target = self._update_target_ae(self.params_linear_target, self.params_linear)
            self.params_critic_target = self._update_target(self.params_critic_target, self.params_critic)

        if writer and self.learning_step % 1000 == 0:
            writer.add_scalar("episode/target_q", target.mean(), self.learning_step)
            writer.add_scalar("episode/mean_q", q_val.mean(), self.learning_step)
            writer.add_scalar("episode/done", done.mean(), self.learning_step)
            writer.add_scalar("episode/reward", reward.mean(), self.learning_step)
            writer.add_scalar("loss/critic", loss_critic, self.learning_step)
            writer.add_scalar("loss/actor", loss_actor, self.learning_step)
            writer.add_scalar("loss/ae", loss_ae, self.learning_step)
            writer.add_scalar("loss/alpha", loss_alpha, self.learning_step)
            writer.add_scalar("stat/alpha", jnp.exp(self.log_alpha), self.learning_step)
            writer.add_scalar("stat/entropy", -mean_log_pi, self.learning_step)

    @partial(jax.jit, static_argnums=0)
    def _calculate_value_list(
        self,
        params_critic: hk.Params,
        last_conv: np.ndarray,
        action: np.ndarray,
    ) -> List[jnp.ndarray]:
        #feature = self.linear.apply(params_critic["linear"], last_conv)
        #return self.critic.apply(params_critic["critic"], feature, action)
        return self.critic.apply(params_critic["critic"], last_conv, action)

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
        last_conv = self.encoder.apply(params_critic["encoder"], state)
        next_last_conv = jax.lax.stop_gradient(self.encoder.apply(params_critic["encoder"], next_state))
        feature = self.linear.apply(params_critic["linear"], last_conv)
        next_feature = jax.lax.stop_gradient(self.linear.apply(params_critic["linear"], next_last_conv))
        return super(SAC_AE, self)._loss_critic(
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
        last_conv = jax.lax.stop_gradient(self.encoder.apply(params_critic["encoder"], state)) #actor update do not flow into AE
        feature = jax.lax.stop_gradient(self.linear.apply(self.params_linear, last_conv))
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
        params_ae: hk.Params,
        state: np.ndarray,
        key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Preprocess states.
        target = preprocess_state(state, key)
        # Reconstruct states.
        last_conv = self.encoder.apply(params_ae["encoder"], state)
        feature = self.linear.apply(params_ae["linear"], last_conv)
        reconst = self.decoder.apply(params_ae["decoder"], feature)
        # MSE for reconstruction errors.
        loss_reconst = jnp.square(target - reconst).mean()
        # L2 penalty of latent representations following RAE.
        loss_latent = 0.5 * jnp.square(feature).sum(axis=1).mean()
        # Weight decay for the decoder.
        loss_weight = weight_decay(params_ae["decoder"])
        return loss_reconst + self.lambda_latent * loss_latent + self.lambda_weight * loss_weight, None

    @property
    def params_ae(self):
        return {
            "encoder": self.params_encoder,
            "linear": self.params_linear,
            "decoder": self.params_decoder,
        }

    @property
    def params_entire_critic(self):
        return {
            "encoder": self.params_encoder,
            "linear": self.params_linear,
            "critic": self.params_critic,
        }

    @property
    def params_entire_critic_target(self):
        return {
            "encoder": self.params_encoder_target,
            "linear": self.params_linear_target,
            "critic": self.params_critic_target,
        }

    def save_params(self, save_dir):
        super().save_params(save_dir)
        save_params(self.params_encoder, os.path.join(save_dir, "params_encoder.npz"))
        save_params(self.params_linear, os.path.join(save_dir, "params_linear.npz"))
        save_params(self.params_decoder, os.path.join(save_dir, "params_decoder.npz"))

    def load_params(self, save_dir):
        super().load_params(save_dir)
        self.params_encoder = self.params_encoder_target = load_params(os.path.join(save_dir, "params_encoder.npz"))
        self.params_linear = self.params_linear_target = load_params(os.path.join(save_dir, "params_linear.npz"))
        self.params_decoder = load_params(os.path.join(save_dir, "params_decoder.npz"))
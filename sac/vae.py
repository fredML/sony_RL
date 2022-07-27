import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

class DownBlock(hk.Module):
    def __init__(self, filter_size):
        super().__init__()
        self.filter_size = filter_size

    def __call__(self, s, is_training):
        #w_init = hk.initializers.Orthogonal(scale=jnp.sqrt(2))
        s = hk.Conv2D(self.filter_size, kernel_shape=3, stride=2)(s)
        s = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.99)(s, is_training=is_training)
        s = jax.nn.leaky_relu(s, 0.01)

        return s

class UpBlock(hk.Module):
    def __init__(self, filter_size):
        super().__init__()
        self.filter_size = filter_size

    def __call__(self, s, is_training):
        #w_init = hk.initializers.Orthogonal(scale=jnp.sqrt(2))
        s = hk.Conv2DTranspose(self.filter_size//2, kernel_shape=3, stride=2)(s)
        s = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.99)(s, is_training=is_training)
        s = jax.nn.leaky_relu(s, 0.01)

        return s

class UpBlockNoBN(hk.Module):
    def __init__(self, filter_size):
        super().__init__()
        self.filter_size = filter_size

    def __call__(self, s):
        #w_init = hk.initializers.Orthogonal(scale=jnp.sqrt(2))
        s = hk.Conv2DTranspose(self.filter_size//2, kernel_shape=3, stride=2)(s)
        s = jax.nn.leaky_relu(s, 0.01)

        return s

class Encoder(hk.Module):
    def __init__(self, latent_dim, filter_sizes):
        super().__init__()
        self.filter_sizes = filter_sizes
        self.latent_dim = latent_dim
    
    def __call__(self, s, is_training):
        for filter_size in self.filter_sizes:
            s = DownBlock(filter_size)(s, is_training)
        s = hk.Flatten()(s)
        mu = hk.Linear(self.latent_dim)(s)
        log_var = hk.Linear(self.latent_dim)(s)
        return mu, log_var

class Encoder_AE(hk.Module):
    def __init__(self, latent_dim, filter_sizes):
        super().__init__()
        self.filter_sizes = filter_sizes
        self.latent_dim = latent_dim
    
    def __call__(self, s, is_training):
        for filter_size in self.filter_sizes:
            s = DownBlock(filter_size)(s, is_training)
        s = hk.Flatten()(s)
        s = hk.Linear(self.latent_dim)(s)
        
        return s

class Decoder(hk.Module):
    def __init__(self, last_conv_shape, filter_sizes, output_channels, final_activation):
        super().__init__()
        self.filter_sizes = filter_sizes
        self.last_conv_shape = last_conv_shape
        #self.n = self.last_conv_shape*self.last_conv_shape*self.filter_sizes[0]
        #self.n = jnp.prod(jnp.array(last_conv_shape))
        self.n = last_conv_shape[0]*last_conv_shape[1]*last_conv_shape[2] #can't use jnp prod in jit for some unknown reason
        self.output_channels = output_channels
        self.final_activation = final_activation
    
    def __call__(self, s, is_training):
        s = jax.nn.leaky_relu(s, 0.01)
        s = hk.Linear(self.n)(s)
        #s = s.reshape(-1, self.last_conv_shape, self.last_conv_shape, self.filter_sizes[0])
        s = s.reshape(-1,*self.last_conv_shape)
        for filter_size in self.filter_sizes:
            s = UpBlock(filter_size)(s, is_training)
        s = hk.Conv2D(self.output_channels, kernel_shape=3, padding='SAME')(s)
        return self.final_activation(s)

class DecoderNoBN(hk.Module):
    def __init__(self, last_conv_shape, filter_sizes, output_channels, final_activation):
        super().__init__()
        self.filter_sizes = filter_sizes
        self.last_conv_shape = last_conv_shape
        self.n = last_conv_shape[0]*last_conv_shape[1]*last_conv_shape[2]
        self.output_channels = output_channels
        self.final_activation = final_activation
    
    def __call__(self, s, is_training): #is_training arg is used for compatibility
        s = jax.nn.leaky_relu(s, 0.01)
        s = hk.Linear(self.n)(s)
        s = s.reshape(-1,*self.last_conv_shape)
        for filter_size in self.filter_sizes:
            s = UpBlockNoBN(filter_size)(s)
        s = hk.Conv2D(self.output_channels, kernel_shape=3, padding='SAME')(s)
        return self.final_activation(s)

class AE(hk.Module):
    def __init__(self, input_size, latent_dim, filter_sizes, output_channels, final_activation):
        super().__init__()
        self.encoder = Encoder_AE(latent_dim, filter_sizes)
        n = len(filter_sizes)
        last_conv_shape = input_size//2**n
        self.decoder = Decoder(last_conv_shape, filter_sizes[::-1], output_channels, final_activation)

    def __call__(self, s, is_training):
        latent = self.encoder(s, is_training)

        recons = self.decoder(latent, is_training)

        return recons, latent, None, None

class VAE(hk.Module): 

    def __init__(self, input_size, latent_dim, filter_sizes, output_channels, final_activation):
        super().__init__()
        self.encoder = Encoder(latent_dim, filter_sizes)
        n = len(filter_sizes)
        last_conv_shape = input_size//2**n
        self.decoder = Decoder(last_conv_shape, filter_sizes[::-1], output_channels, final_activation)

    def __call__(self, s, is_training):
        mu, log_var = self.encoder(s, is_training)
        sigma = jnp.exp(0.5*log_var)*is_training
        latent = mu + np.random.normal(0, 1, size=sigma.shape)*sigma

        recons = self.decoder(latent, is_training)

        return recons, latent, mu, log_var

class ResNetVAE(hk.Module):
    def __init__(self, latent_dim, filter_sizes, weights, pooling, output_channels, final_activation):
        import os
        path = os.getcwd()
        os.chdir('/mnt/diskSustainability/frederic/haikumodels') 
        import haikumodels as hm
        os.chdir(path)

        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = hm.ResNet50(include_top=False, weights=weights, pooling=pooling)
        self.decoder = DecoderNoBN(filter_sizes[::-1], output_channels, final_activation)
        self.enc_flattened_shape = 4*4*2048
    
    def __call__(self, s, encoder_training, encoder_bn_training):
        if encoder_training:
            enc_output = self.encoder(s, encoder_bn_training)
        else:
            enc_output = jax.lax.stop_gradient(self.encoder(s, encoder_bn_training))

        h = hk.Flatten()(enc_output)
        mu = hk.Linear(self.latent_dim)(h)
        log_var = hk.Linear(self.latent_dim)(h)
        sigma = jnp.exp(0.5*log_var)*encoder_training
        latent = mu + np.random.normal(0, 1, size=sigma.shape)*sigma

        recons = self.decoder(latent, encoder_bn_training)

        return recons, latent, mu, log_var

def kl_gaussian(mean: jnp.ndarray, log_var: jnp.ndarray) -> jnp.ndarray:

  return 0.5 * jnp.sum(-log_var - 1.0 + jnp.exp(log_var) + jnp.square(mean), axis=-1).mean()

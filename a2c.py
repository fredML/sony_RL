from sony_RL.base_functions.dm_env_sphere import SphereEnv
from sony_RL.sac import vae
from sony_RL.sac.sac import SAC
from sony_RL.sac.base_trainer import Trainer
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import os

import warnings
warnings.filterwarnings("ignore")

##

gt = np.load('/mnt/diskSustainability/frederic/scanner-gym_models_v2/sphere_hole/ground_truth_volumes/gt_0.npy')

x,y,z = np.where(gt==-1)

mask = ((y==12) & (z==12)) | ((y==12) & (z==13)) | ((y==13) & (z==12)) | ((y==13) & (z==13))
mask = mask & (x>4) & (x<21)
mask1 = ((y==z) & (x==12)) | ((y==z) & (x==13))
mask1 = mask1 & (y>4) & (y<21)
mask2 = ((y==25-z) & (x==12)) | ((y==25-z) & (x==13))
mask2 = mask2 & (y>4) & (y<21)

mask = mask | mask1 | mask2

voxel_weights = np.zeros((25,25,25))

for i in range (len(x[mask])):
    voxel_weights[x[mask][i],y[mask][i],z[mask][i]] = 1

objects_path = '/mnt/diskSustainability/frederic/scanner-gym_models_v2/sphere_hole'
env = SphereEnv(objects_path, img_shape=128, voxel_weights=voxel_weights, rmax=0.5, mode='eval')
env_test = SphereEnv(objects_path, img_shape=128, voxel_weights=voxel_weights, rmax=0.5, mode='eval')
ts = env.reset()

##

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'

input_size = 128
num_epochs = 5000
batch_size = 16
seed = np.random.randint(100)
filter_sizes = [16,32,64,128]
input_channels = output_channels = 1
final_activation = jax.jit(lambda s:s)

latent_dim = 14
lambda_kl = 9.21e-5

def classic_vae(s, is_training):
    return vae.VAE(input_size, latent_dim, filter_sizes, output_channels, final_activation, coord_conv=True)(s, is_training)

print('##### VAE initialization #####')

vae_init, vae_apply  = hk.without_apply_rng(hk.transform_with_state(classic_vae))
vae_apply_jit = jax.jit(vae_apply, static_argnums=3)

weights = jnp.load('/mnt/diskSustainability/frederic/sony_RL/params_vae_lat=14_kl=9.21e-05.npz', allow_pickle=True)
params_vae = weights['params_vae'][()]
bn_vae_state =  weights['bn_vae_state'][()]

print()
print('##### Initialization finished #####')

##

seed = np.random.randint(100)
print()
print('seed = {}'.format(seed))
print()
print('##### Agent initialization #####')
encoder = (vae_apply_jit, params_vae, bn_vae_state)
agent = SAC(num_agent_steps=10**6, state_space=np.empty(env.observation_shape), action_space=np.empty(2), 
           seed=seed, start_steps=10**3, gamma=0.7, buffer_size=10**3, batch_size=32, encoder=encoder)

print()
print('##### Initialization finished #####')
print()
print('##### Training RL agent #####')

trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=agent,
        log_dir='sac_beta_coordconv_vae_logs',
        num_agent_steps=10**6,
        action_repeat=1,
        eval_interval=10**3,
        save_params=True,
        save_interval=10**4
    )

trainer.train()

print()
print('##### Training finished #####')
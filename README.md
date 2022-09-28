# sony_RL

This repo contains all the work pursued at Sony CSL Paris as a research intern for 3D reconstruction using Deep Reinforcement Learning. It is still very experimental and is not packaged yet. 
Functions for the space carving, RL environment, RL agents are given. Only JAX/Haiku is used for training neural networks. 

Old base functions folder contains .py files inspired from the previous framework found in RL_NBV github. 
Base functions folder contains my own implementations which are more efficient. In a nutshell, important functions are dm_env_sphere.py, space_carving.py, cl.py, utils.py. 
The rl folder is essential and contains all offline RL agents considered + vae implementations

Library rlviewer must be installed to use continuous agents for fast object rendering. See https://github.com/romi/rlviewer

There is a rl_with_batch_norm branch which adds batch norm layers to the RL agents but has been not been updated in a while so it is unsure if it works.

To use Blender Python, download a version of Blender. Open an xorg server and create a fake screen with "sh vscanner_launch.sh"

##### 

Having some technical issues with packaging, I give here notebooks instead of .py files for running experiments

########

There are two notebooks for solving the 3D reconstruction problem using DRL on simple geometric objects which are pierced spheres (and an attempt of plant creation).

The long notebook is separated into several blocks of code including: objects creation, environment initialization, VAE pretraining (only the classic one has been tuned for disantanglement metric, not VQVAE and CATVAE yet), UMAP for visualization, RL training.

The short one supposes that the weights of the VAE are available (change path if needed) and directly trains a RL agent. It can be run with shell command "ipython3 -c '%run rl_3d_recons_short.ipynb'".

# sony_RL

This repo contains all the work pursued at Sony CSL Paris as a research intern for 3D reconstruction using Deep Reinforcement Learning.
It is still very experimental and is not packaged yet. Hence experiments and 3D objects used are only available on notebooks which are not on this repo.
Only functions for the space carving, RL environment, RL agents are given. Only JAX/Haiku is used for training neural networks. 

Lots of base functions are useless and are kept for work timeline but should be removed. This includes folder a2c which contains old inefiicient 
implementations of rl. In a nutshell, important base functions are dm_env_sphere.py, space_carving.py, cl.py, utils.py. 
The sac folder is essential and contains all offline RL agents considered.

Library rlviewer must be installed to use continuous agents for fast object rendering

There is a rl_with_batch_norm branch which adds batch norm layers to the RL agents.

To use Blender Python, download a version of Blender. Open an xorg server and create a fake screen with "sh vscanner_launch.sh"

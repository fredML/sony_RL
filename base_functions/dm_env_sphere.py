import numpy as np
from space_carving import *
import dm_env
from acme import specs
from utils import angle_to_position_continuous
from skimage.transform import resize


class SphereEnv(dm_env.Environment):

    def __init__(self, objects_path, voxel_weights=None, rmax=0.9, k_d=0, k_t=0, img_shape=84):

        super(SphereEnv, self).__init__()
        self.objects_path = objects_path
        self.voxel_weights = voxel_weights
        self.spc = space_carving_rotation_2d(self.objects_path,
                                             voxel_weights=self.voxel_weights,
                                             continuous=True)
        self.max_reward = rmax
        self.k_d = k_d #penalty based on distance between two camera positions
        self.k_t = k_t #penalty based on number of views

        self.env_shape = (img_shape,img_shape,3)

    def reset(self, theta_init=-1, phi_init=-1) -> dm_env.TimeStep:
        self.num_steps = 0
        self.total_reward = 0

        # keep track of visited positions during the episode
        self.visited_positions = []

        # inital position of the camera, if -1 choose random
        self.init_phi = phi_init
        self.init_theta = theta_init

        if self.init_phi == -1:
            self.init_phi = np.random.uniform(0, 2*np.pi)
            self.current_phi = self.init_phi
        else:
            self.current_phi = self.init_phi

        if self.init_theta == -1:
            self.init_theta = np.random.uniform(0, np.pi/2)
            self.current_theta = self.init_theta
        else:
            self.current_theta = self.init_theta

        # append initial position to visited positions list
        self.visited_positions.append([self.current_theta, self.current_phi])

        # create space carving objects
        self.spc.reset()

        pos = angle_to_position_continuous(
            self.current_theta, self.current_phi)
        img = self.spc.get_image_continuous(self.current_theta, self.current_phi)
        img_gray = np.dot(img, [0.2989, 0.5870, 0.1140])
        img_gray = resize(img_gray,self.env_shape[:2])
        self.last_k_pos = np.concatenate((pos, pos, pos))
        self.last_k_img = np.stack((img_gray,img_gray,img_gray), axis=-1)
        #observation = self.last_k_pos.astype('float32')
        observation = self.last_k_img.astype('float32')

        return dm_env.restart(observation)

    def step(self, action) -> dm_env.TimeStep:
        theta, phi = action
        self.num_steps += 1
        self.current_theta += theta
        if self.current_theta > np.pi/2:
            self.current_theta -= np.pi/2
        elif self.current_theta < 0:
            self.current_theta += np.pi/2

        self.current_phi += phi
        if self.current_phi >= 2*np.pi:
            self.current_phi -= 2*np.pi
        elif self.current_phi < 0:
            self.current_phi += 2*np.pi

        self.visited_positions.append([self.current_theta, self.current_phi])

        self.spc.continuous_carve(self.current_theta, self.current_phi)

        self.reward = self.spc.gt_compare()

        pos = angle_to_position_continuous(
            self.current_theta, self.current_phi)
        img = self.spc.img
        img_gray = np.dot(img, [0.2989, 0.5870, 0.1140])
        img_gray = resize(img_gray,self.env_shape[:2])
        self.penalty = np.linalg.norm(pos-self.last_k_pos[-3:])
        self.last_k_pos = np.concatenate((self.last_k_pos[3:], pos))
        self.last_k_img = np.concatenate((self.last_k_img[...,1:], img_gray[...,None]), axis=-1)

        self.total_reward += self.reward
        if self.reward > 0:
            self.reward -= self.k_d*self.penalty
            self.reward = max(self.reward,0)
        else:
            self.reward = -self.k_t

        #observation = self.last_k_pos.astype('float32')
        observation = self.last_k_img.astype('float32')

        if self.total_reward > self.max_reward:
            return dm_env.termination(reward=self.reward*1., observation=observation)

        return dm_env.transition(reward=self.reward*1., observation=observation)

    def step_angle(self, theta, phi) -> dm_env.TimeStep:

        self.num_steps += 1
        self.current_theta = theta
        self.current_phi = phi

        self.visited_positions.append([self.current_theta, self.current_phi])

        self.spc.continuous_carve(self.current_theta, self.current_phi)

        self.reward = self.spc.gt_compare()

        pos = angle_to_position_continuous(
            self.current_theta, self.current_phi)
        img = self.spc.img
        img_gray = np.dot(img, [0.2989, 0.5870, 0.1140])
        img_gray = resize(img_gray,self.env_shape[:2])
        self.penalty = np.linalg.norm(pos-self.last_k_pos[-3:])
        self.last_k_pos = np.concatenate((self.last_k_pos[3:], pos))
        self.last_k_img = np.concatenate((self.last_k_img[...,1:], img_gray[...,None]), axis=-1)

        self.total_reward += self.reward
        if self.reward > 0:
            self.reward -= self.k_d*self.penalty
        else:
            self.reward = -self.k_t

        #observation = self.last_k_pos.astype('float32')
        observation = self.last_k_img.astype('float32')

        if self.total_reward > self.max_reward:
            return dm_env.termination(reward=self.reward*1., observation=observation)

        return dm_env.transition(reward=self.reward*1., observation=observation)

    def observation_spec(self) -> specs.BoundedArray:
        if len(self.env_shape) == 3:
            return specs.BoundedArray(shape=self.env_shape, minimum=0, maximum=255, dtype=np.uint8)
        return specs.BoundedArray(shape=(9), minimum=-1, maximum=1, dtype=np.float32)

    def action_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray(shape=(2,), minimum=-1, maximum=1, dtype=np.float32)

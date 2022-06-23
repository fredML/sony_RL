import numpy as np
from space_carving import *
import dm_env
from acme import specs
from utils import angle_to_position_continuous


class SphereEnv(dm_env.Environment):

    def __init__(self, objects_path, voxel_weights=None):

        super(SphereEnv, self).__init__()
        #self.__version__ = "7.0.1"
        self.objects_path = objects_path
        self.voxel_weights = voxel_weights
        self.spc = space_carving_rotation_2d(self.objects_path,
                                             voxel_weights=self.voxel_weights,
                                             continuous=True)

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
        # carve image from initial position
        '''self.spc.continuous_carve(self.current_theta, self.current_phi)
        self.reward = self.spc.gt_compare()'''

        pos = angle_to_position_continuous(
            self.current_theta, self.current_phi)
        self.last_k_pos = np.concatenate((pos, pos, pos))
        observation = self.last_k_pos

        return dm_env.restart(observation)

    def step(self, action) -> dm_env.TimeStep:
        theta, phi = action
        theta = np.pi/4*theta
        phi = np.pi*phi
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
        self.last_k_pos = np.concatenate((self.last_k_pos[3:], pos))

        self.total_reward += self.reward
        observation = self.last_k_pos

        if self.total_reward > 0.3:
            return dm_env.termination(reward=self.reward, observation=observation)

        return dm_env.transition(reward=self.reward, observation=observation)

    def step_angle(self, theta, phi) -> dm_env.TimeStep:

        self.num_steps += 1
        self.current_theta = theta
        self.current_phi = phi

        self.visited_positions.append([self.current_theta, self.current_phi])

        self.spc.continuous_carve(self.current_theta, self.current_phi)

        self.reward = self.spc.gt_compare()

        pos = angle_to_position_continuous(
            self.current_theta, self.current_phi)
        self.last_k_pos = np.concatenate((self.last_k_pos[3:], pos))

        self.total_reward += self.reward
        observation = self.last_k_pos

        if self.total_reward > 0.3:
            dm_env.termination(reward=self.reward, observation=observation)

        return dm_env.transition(reward=self.reward, observation=observation)

    def observation_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray(shape=(9,), minimum=-1., maximum=1., dtype=np.float32)

    def action_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray(shape=(2,), minimum=-[np.pi/4, np.pi], maximum=[np.pi/4, np.pi], dtype=np.float32)
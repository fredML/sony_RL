import numpy as np
import os
from space_carving import *
import random
import collections
from math import prod

TimeStep = collections.namedtuple("TimeStep", field_names=["step_type", "reward", "discount", "observation"])

class ScannerEnv():
    """
    Custom OpenAI Gym environment  for training 3d scannner
    """

    def __init__(self, objects_path, train_objects, cube_view='static', voxel_weights = None):

        super(ScannerEnv, self).__init__()
        #self.__version__ = "7.0.1"
        # number of images that must be collected
        self.objects_path = objects_path
        # total of posible positions for phi  in env
        self.phi_n_positions = 180
        # total of posible positions for theta in env
        self.theta_n_positions = 4
        # 3d objects used for training
        self.train_objects = train_objects
        # if static, figure in cube is always seen from the same perspective
        # if dynamic, position figure in cube rotates according to the camera perspective
        self.cube_view = cube_view

        # images
        self.images_shape = (84, 84, 3)

        # map action with correspondent movements in phi and theta
        # phi numbers are number of steps relative to current position
        # theta numbers are absolute position in theta
        # (phi,theta)

        self.actions = {}
        compt = 0
        for k in range (4):
            for j in range (-15,20,5):
                self.actions[compt]=(j,k)
                compt += 1
        
        self.discount = 0.95
        self.voxel_weights = voxel_weights
        self.num_actions = len(self.actions)

    def reset(self, phi_init=-1, theta_init=-1, phi_bias=0):
        self.num_steps = 0
        self.total_reward = 0
        self.done = False

        # keep track of visited positions during the episode
        self.visited_positions = []

        # inital position of the camera, if -1 choose random
        self.init_phi = phi_init
        self.init_theta = theta_init

        if self.init_phi == -1:
            self.init_phi = np.random.randint(0, self.phi_n_positions)
            self.current_phi = self.init_phi
        else:
            self.current_phi = self.init_phi

        if self.init_theta == -1:
            self.init_theta = np.random.randint(0, self.theta_n_positions)
            self.current_theta = self.init_theta
        else:
            self.current_theta = self.init_theta

        # simulates rotation of objects (z axis) by n steps (for data augmentation), -1 for random rotation
        if phi_bias == -1:
            self.phi_bias = np.random.randint(0, self.phi_n_positions)
        else:
            self.phi_bias = phi_bias

        # append initial position to visited positions list
        self.visited_positions.append((self.current_theta, self.current_phi))

        # take random  model from available objects list
        object = random.choice(self.train_objects)
        self.current_object = object

        # create space carving objects
        self.spc = space_carving_rotation_2d(os.path.join(self.objects_path, object),
                                             phi_bias=self.phi_bias,
                                             total_phi_positions=self.phi_n_positions,
                                             cube_view='static', voxel_weights=self.voxel_weights)
        # carve image from initial position
        self.spc.carve(self.current_phi, self.current_theta)
        vol = self.spc.last_volume

        self.volume_shape = vol.shape
        self.last_gt_ratio = 0

        # get camera image
        self.im3 = np.array(self.spc.get_image(self.current_phi, self.current_theta))[...,:3]/255
        #im = np.array(self.spc.get_image(self.current_phi, self.current_theta))
        # need 3 last images, this is first taken image so copy it 3 times
        # and adjust dimensions (height,width,channel)
        #self.im3 = np.stack([im, im, im],axis=-1)

        gt_ratio = self.spc.gt_compare()
        self.last_gt_ratio = gt_ratio
        self.reward = gt_ratio
                
        pos = np.array([np.cos(self.current_phi*np.pi/180)*np.sin((self.current_theta+1)*np.pi/8),
                        np.sin(self.current_phi*np.pi/180)*np.sin((self.current_theta+1)*np.pi/8),
                        np.cos((self.current_theta+1)*np.pi/8)])

        self.last_k_pos = np.concatenate((pos,pos,pos))
        observation = self.last_k_pos

        return TimeStep(step_type=False, reward=self.reward, discount=self.discount, observation=observation)

    
    def step(self, action):
        self.num_steps += 1

        phi = self.actions[action][0]
        theta = self.actions[action][1]

        # move n phi steps from current phi position
        self.current_phi = self.calculate_phi_position(
            self.current_phi, phi)
        # theta indicates absolute position
        # check theta limits
        if theta < 0:
            theta = 0
        elif theta >= self.theta_n_positions:
            theta = self.theta_n_positions-1

        self.current_theta = theta

        self.visited_positions.append([self.current_theta,self.current_phi])

        # carve in new position
        self.spc.carve(self.current_phi, self.current_theta)
        vol = self.spc.last_volume

        # get camera image
        self.im3 = np.array(self.spc.get_image(self.current_phi, self.current_theta))[...,:3]/255
        #im = np.array(self.spc.get_image(self.current_phi, self.current_theta))
        # need 3 last images, #and adjust dimensions (height,width,channel)
        #self.im3 = np.stack([self.im3[:, :, 1], self.im3[:, :, 2], im], axis=-1)

        #calculate increment of solid voxels ratios between gt and current volume
        '''gt_ratio = self.spc.gt_compare()
        delta_gt_ratio = gt_ratio - self.last_gt_ratio
        self.last_gt_ratio = gt_ratio
        self.reward = delta_gt_ratio'''
        self.reward = self.spc.gt_compare()

        pos = np.array([np.cos(self.current_phi*np.pi/180)*np.sin((self.current_theta+1)*np.pi/8),
                        np.sin(self.current_phi*np.pi/180)*np.sin((self.current_theta+1)*np.pi/8),
                        np.cos((self.current_theta+1)*np.pi/8)])

        #dist_penalty = np.linalg.norm(pos-self.last_pos)

        self.last_k_pos = np.concatenate((self.last_k_pos[3:],pos))

        self.total_reward += self.reward
        observation = self.last_k_pos

        if self.total_reward > 0.3:
            self.done = True

        return TimeStep(step_type=self.done, reward=self.reward, discount=self.discount, observation=observation)


    def step_angle(self, theta, phi):
        self.num_steps += 1
        self.current_phi = phi
        self.current_theta = theta
        self.visited_positions.append([self.current_phi, self.current_theta]) 

        self.spc.carve(self.current_phi, self.current_theta)
        vol = self.spc.last_volume

        im = np.array(self.spc.get_image(self.current_phi, self.current_theta))
        self.im3 = np.stack([self.im3[:, :, 1], self.im3[:, :, 2], im], axis=-1)

        if self.gt_mode:
            gt_ratio = self.spc.gt_compare()
            delta_gt_ratio = gt_ratio - self.last_gt_ratio
            self.last_gt_ratio = gt_ratio
            reward = delta_gt_ratio
        
        else:
            _, self.voxel_count = np.unique(vol,return_counts=True)
            reward =  (self.last_voxel_count - self.voxel_count)[1]/prod(self.volume_shape)

            self.last_voxel_count = self.voxel_count

        self.total_reward += reward

        self.current_state = (self.im3,
                              vol.astype('float32')[...,None])

        return TimeStep(step_type=self.done, reward=reward, discount=self.discount, observation=self.im3/255)

    def generate_gt(self):
        for theta in range(self.theta_n_positions):
            for phi in range(self.phi_n_positions):
                self.current_theta = theta
                self.current_phi = phi
                self.spc.carve(self.current_phi, self.current_theta)
        return self.spc.volume

    def minMaxNorm(self, old, oldmin, oldmax, newmin, newmax):
        return ((old-oldmin)*(newmax-newmin)/(oldmax-oldmin)) + newmin

    def phi_from_continuous(self, cont):
        '''converts float from (-1,1) to int (-90,90) '''
        return int(self.minMaxNorm(cont, -1.0, +1.0, -90, 90))

    def theta_from_continuous(self, cont):
        '''converts float from (-1,1) to int (0,3) '''
        return int(self.minMaxNorm(cont, -1.0, +1.0, 0.0, 3.0))

    def calculate_phi_position(self, curr_phi, steps):
        n_pos = curr_phi + steps
        if n_pos > (self.phi_n_positions-1):
            n_pos -= self.phi_n_positions
        elif n_pos < 0:
            n_pos += self.phi_n_positions
        return n_pos



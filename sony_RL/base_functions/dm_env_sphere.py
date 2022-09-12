import numpy as np
from space_carving import *
import dm_env
from acme import specs
from utils import angle_to_position_continuous
import cv2 as cv
from PIL import Image

class SphereEnv(dm_env.Environment):

    def __init__(
        self, 
        objects_path, 
        objects_name,
        img_shape, 
        use_img=True,
        use_goal=False, 
        continuous=True, 
        last_k=3, 
        voxel_weights=None, 
        list_holes=None,
        rmax_T=1, 
        k_t=0, 
        mode='eval', 
        max_T=50,
        phi_n_positions=180,
        theta_n_positions=4):

        super(SphereEnv, self).__init__()
        self.objects_path = objects_path
        self.objects_name = objects_name
        self.voxel_weights = voxel_weights
        self.list_holes = list_holes
        self.continuous = continuous

        self.rmax_T = rmax_T
        self.img_shape = img_shape # for reshaping images
        self.last_k = last_k
        self.max_T = max_T

        self.mode = mode
        self.k_t = k_t #penalty based on number of views
        self.use_img = use_img
        self.use_goal = use_goal

        self.phi_n_positions = phi_n_positions
        self.theta_n_positions = theta_n_positions

        self.actions = {}
        compt = 0
        for k in range (theta_n_positions):
            for j in range (-15, 20, 5):
                self.actions[compt] = (k, j)
                compt += 1

    def reset(self, obj=None, theta_init=-1, phi_init=-1) -> dm_env.TimeStep:
        self.num_steps = 0
        self.total_reward = 0
        self.done = False

        # keep track of visited positions during the episode
        self.visited_positions = []

        # inital position of the camera, if -1 choose random
        self.init_phi = phi_init
        self.init_theta = theta_init

        if self.init_phi == -1:
            if self.continuous:
                self.init_phi = np.random.uniform(0, 2*np.pi)
            else:
                self.init_phi = np.random.randint(0, self.phi_n_positions)
            self.current_phi = self.init_phi
        else:
            self.current_phi = self.init_phi

        if self.init_theta == -1:
            if self.continuous:
                self.init_theta = np.random.uniform(0, np.pi/2)
            else:
                self.init_theta = np.random.randint(0, self.theta_n_positions)
            self.current_theta = self.init_theta
        else:
            self.current_theta = self.init_theta

        # append initial position to visited positions list
        self.visited_positions.append([self.current_theta, self.current_phi])

        # create space carving objects
        if obj is None:
            obj = np.random.choice(len(self.objects_name))
        if self.voxel_weights is None:
            self.current_spc = space_carving_rotation_2d(
                                                self.objects_path[obj], 
                                                self.objects_name[obj],
                                                list_holes=self.list_holes[obj],
                                                continuous=self.continuous)
        else:        
            self.current_spc = space_carving_rotation_2d(
                                                self.objects_path[obj], 
                                                self.objects_name[obj],
                                                voxel_weights=self.voxel_weights[obj],
                                                continuous=self.continuous)
        self.current_obj = obj
        self.current_spc.reset()
        self.current_rmax_inf = self.current_spc.rmax_inf
        self.n_goals = self.remaining_goals = len(self.list_holes[obj])*1.

        if self.use_img:
            if self.continuous:
                img = self.current_spc.get_image_continuous(self.current_theta, self.current_phi)
            else:
                img = self.current_spc.get_image(self.current_theta, self.current_phi)

            pil_img = Image.fromarray(img).resize((self.img_shape, self.img_shape), Image.Resampling.NEAREST)
            self.img = np.array(pil_img)
            img_gray = np.array(pil_img.convert('L'))
            self.img_gray = img_gray[...,None]

            canny = cv.Canny(img_gray, 40, 80)[...,None] 
            self.canny = canny 
                        
            self.last_k_img = np.stack([self.canny]*self.last_k, axis=0) # shape (k,img_size,img_size,1)
            self.observation = self.last_k_img
            if self.use_goal:
                self.remaining_goals_img = self.remaining_goals*np.ones_like(canny[None])
                self.observation = np.concatenate((self.observation, np.tanh(self.remaining_goals_img)))
        
        else:
            if self.continuous:
                pos = angle_to_position_continuous(self.current_theta, self.current_phi)
            else:
                pos = angle_to_position_continuous((self.current_theta+1)*np.pi/(2*self.theta_n_positions),
                                                    self.current_phi*2*np.pi/self.phi_n_positions)
            self.pos = pos
            self.last_k_pos = np.concatenate([pos]*self.last_k)
            self.observation = self.last_k_pos

        self.observation_shape = self.observation.shape

        return dm_env.restart(self.observation*1.)

    def _step(self):

        self.visited_positions.append([self.current_theta, self.current_phi])
        if self.continuous:
            self.current_spc.continuous_carve(self.current_theta, self.current_phi)
        else:
            self.current_spc.carve(self.current_theta, self.current_phi)

        self.reward = self.current_spc.gt_compare()
        if self.current_rmax_inf:
            self.reward = self.reward/self.current_rmax_inf
        self.total_reward += self.reward

        self.remaining_goals = int(self.n_goals - self.total_reward//(1/self.n_goals))

        if self.reward == 0:
            if self.mode == 'train':
                self.reward -= self.k_t

        if self.use_img:
            if self.continuous:
                img = self.current_spc.img
            else:
                img = self.current_spc.get_image(self.current_theta, self.current_phi)

            pil_img = Image.fromarray(img).resize((self.img_shape, self.img_shape), Image.Resampling.NEAREST)
            self.img = np.array(pil_img)
            img_gray = np.array(pil_img.convert('L'))
            self.img_gray = img_gray[...,None]

            canny = cv.Canny(img_gray, 40, 80)[...,None] 
            self.canny = canny

            if self.use_goal:
                temp = np.empty(self.observation_shape)[1:]
            else:
                temp = np.empty(self.observation_shape)
            temp[:2] = self.last_k_img[1:]
            temp[2] = canny[None]
            self.last_k_img = temp
            self.observation = self.last_k_img
            if self.use_goal:
                self.remaining_goals_img = self.remaining_goals*np.ones_like(canny[None])
                self.observation = np.concatenate((self.observation, np.tanh(self.remaining_goals_img)))
        
        else:
            if self.continuous:
                pos = angle_to_position_continuous(self.current_theta, self.current_phi)
            else:
                pos = angle_to_position_continuous((self.current_theta+1)*np.pi/(2*self.theta_n_positions),
                                                    self.current_phi*2*np.pi/self.phi_n_positions)
                
            self.pos = pos
            self.last_k_pos = np.concatenate((self.last_k_pos[3:], pos))
            self.observation = self.last_k_pos

        if self.total_reward > self.rmax_T:
            self.done = True
            return dm_env.termination(reward=self.reward, observation=self.observation*1.)
        
        if (self.use_goal == True) & (self.remaining_goals == 0):
            self.done = True
            return dm_env.termination(reward=self.reward, observation=self.observation*1.)

        if self.num_steps > self.max_T: # for "non-environmental" time limits, it might be better to still bootstrap
            ts = dm_env.transition(reward=self.reward, observation=self.observation*1.)
            self.reset()
            return ts

        return dm_env.transition(reward=self.reward, observation=self.observation)


    def step(self, action) -> dm_env.TimeStep:
        self.num_steps += 1
        if self.continuous:
            theta, phi = action
            self.current_theta = self.calculate_angle(self.current_theta, np.pi, theta)
            self.current_phi = self.calculate_angle(self.current_phi, 2*np.pi, phi)
        else:
            action = action.item()
            if isinstance(action, float): # add this cond for actions obtained by randomization (by default a float)
                action = np.random.randint(len(self.actions))
            theta, phi = self.actions[action]
            self.current_theta = theta
            self.current_phi = self.calculate_angle(self.current_phi, self.phi_n_positions, phi)

        return self._step()

    def step_angle(self, theta, phi) -> dm_env.TimeStep:

        self.num_steps += 1
        self.current_theta = theta
        self.current_phi = phi

        return self._step()

    def calculate_angle(self, curr_angle, max_angle, steps):
        n_pos = curr_angle + steps
        if n_pos >= max_angle:
            n_pos -= max_angle
        elif n_pos < 0:
            n_pos += max_angle
        return n_pos

    '''def segmentation_mask(self):

        neigh_xyz = []
        dtheta = 6*np.pi/180
        dphi = 6*np.pi/180
        for i in range(len(self.opt_theta)):
            theta, phi = self.opt_theta[i], self.opt_phi[i]
            for val_theta in np.linspace(theta-dtheta,theta+dtheta,50):
                for val_phi in np.linspace(phi-dphi,phi+dphi,50):
                    neigh_xyz.append(self.sphere_radius*angle_to_position_continuous(val_theta, val_phi))
        
        neigh_xyz = np.array(neigh_xyz)
        self.neigh_xyz = neigh_xyz

        segmentation_mask = np.zeros((1024,768))

        distances = np.linalg.norm(self.pos - self.neigh_xyz, axis=1)
        self.distances = distances
        mask = np.where(distances < np.sqrt(self.spc.radius**2 + self.sphere_radius**2))
        self.neigh_xyz = self.neigh_xyz[mask]

        extrinsics = self.spc.extrinsics
        R = np.array(extrinsics['R'])
        T = np.array(extrinsics['T'])

        pixels = R.dot(self.neigh_xyz.T) + T[...,None] #transform world coordinates to camera coordinates
        mask = np.where(pixels[2]>0)
        pixels = pixels[:,mask[0]]

        K = self.spc.intrinsics
        x = K[0]*pixels[0]/pixels[2] + K[2]
        y = K[1]*pixels[1]/pixels[2] + K[3]
        x = x.astype('int32')
        y = y.astype('int32')

        self.pixels = (x,y)

        for i in range (len(x)):
            segmentation_mask[x[i],y[i]] = 1
        
        segmentation_mask = segmentation_mask.T
        pil_img = Image.fromarray(segmentation_mask).resize((self.img_shape, self.img_shape), PIL.Image.Resampling.NEAREST)

        return np.array(pil_img)'''

    def observation_spec(self) -> specs.BoundedArray:
        '''if (len(self.env_shape) == 3) & (self.apply_ae is None):
            return specs.BoundedArray(shape=self.env_shape, minimum=0, maximum=255, dtype=np.uint8)
        return specs.BoundedArray(shape=self.env_shape, minimum=-1, maximum=1, dtype=np.float32)'''
        pass

    def action_spec(self) -> specs.BoundedArray:
        '''return specs.BoundedArray(shape=(2,), minimum=-1, maximum=1, dtype=np.float32)'''
        pass
import numpy as np
from space_carving import *
import dm_env
from acme import specs
from utils import angle_to_position_continuous
import PIL
import cv2 as cv

class SphereEnv(dm_env.Environment):

    def __init__(self, objects_path, img_shape, last_k = 3, voxel_weights=None, rmax=0.9, k_d=0, k_t=0):

        super(SphereEnv, self).__init__()
        self.objects_path = objects_path
        self.voxel_weights = voxel_weights
        self.spc = space_carving_rotation_2d(self.objects_path,
                                             voxel_weights=self.voxel_weights,
                                             continuous=True)
        self.max_reward = rmax
        self.k_d = k_d #penalty based on distance between two camera positions
        self.k_t = k_t #penalty based on number of views

        self.sphere_radius = 5
        self.opt_theta = np.pi*np.array([1/2,1/4,3/4,1/2,1/4,3/4])
        self.opt_phi = np.pi*np.array([0,1/2,1/2,1,-1/2,-1/2])
        self.opt_pos = self.sphere_radius * angle_to_position_continuous(self.opt_theta, self.opt_phi).T

        self.img_shape = img_shape # for reshaping images

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

        self.last_k = last_k

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

        pos = self.spc.radius*angle_to_position_continuous(
            self.current_theta, self.current_phi)
        self.pos = pos
        img = self.spc.get_image_continuous(self.current_theta, self.current_phi)
        pil_img = Image.fromarray(img).resize((self.img_shape, self.img_shape))
        self.img = np.array(pil_img)
        img_gray = np.array(pil_img.convert('L'))
        self.img_gray = img_gray[...,None]

        canny = cv.Canny(img_gray, 40, 80)[...,None] * 1.
        self.canny = canny 
                        
        self.last_k_img = np.stack([self.canny]*self.last_k, axis=0) # shape (k,img_size,img_size,1)
        self.observation = self.last_k_img.astype('float32')

        self.last_k_pos = np.concatenate([pos]*self.last_k)
        #self.observation = self.last_k_pos.astype('float32')
        #self.opt_dist = np.linalg.norm(self.pos - self.opt_pos, axis=1)

        self.observation_shape = self.observation.shape

        return dm_env.restart(self.observation)

    def step(self, action) -> dm_env.TimeStep:
        theta, phi = action
        self.num_steps += 1
        self.current_theta += theta
        if self.current_theta > np.pi:
            self.current_theta -= np.pi
        elif self.current_theta < 0:
            self.current_theta += np.pi

        self.current_phi += phi
        if self.current_phi >= 2*np.pi:
            self.current_phi -= 2*np.pi
        elif self.current_phi < 0:
            self.current_phi += 2*np.pi

        self.visited_positions.append([self.current_theta, self.current_phi])

        self.spc.continuous_carve(self.current_theta, self.current_phi)

        self.reward = self.spc.gt_compare()

        pos = self.spc.radius*angle_to_position_continuous(
            self.current_theta, self.current_phi)
        self.pos = pos
        img = self.spc.img
        pil_img = Image.fromarray(img).resize((self.img_shape, self.img_shape))
        self.img = np.array(pil_img)
        img_gray = np.array(pil_img.convert('L'))
        self.img_gray = img_gray[...,None]
        self.penalty = np.linalg.norm(pos-self.last_k_pos[-3:])
        self.last_k_pos = np.concatenate((self.last_k_pos[3:], pos))

        #self.opt_dist = np.linalg.norm(self.pos - self.opt_pos, axis=1)

        self.total_reward += self.reward
        if self.reward > 0:
            self.reward -= self.k_d*self.penalty
            self.reward = max(self.reward,0)
        else:
            self.reward = -self.k_t

        canny = cv.Canny(img_gray, 40, 80)[...,None] * 1.
        self.canny = canny
                        
        self.last_k_img = np.concatenate((self.last_k_img[1:], self.canny[None]), axis=0) # shape (k,img_size,img_size,1)
        self.observation = self.last_k_img.astype('float32')

        #self.observation = self.last_k_pos.astype('float32')

        self.observation_shape = self.observation.shape

        if self.total_reward > self.max_reward:
            return dm_env.termination(reward=self.reward*1., observation=self.observation)

        return dm_env.transition(reward=self.reward*1., observation=self.observation)

    def step_angle(self, theta, phi) -> dm_env.TimeStep:

        self.num_steps += 1
        self.current_theta = theta
        self.current_phi = phi

        if self.current_theta > np.pi/2:
            self.current_theta -= np.pi/2
        elif self.current_theta < 0:
            self.current_theta += np.pi/2

        if self.current_phi >= 2*np.pi:
            self.current_phi -= 2*np.pi
        elif self.current_phi < 0:
            self.current_phi += 2*np.pi

        self.visited_positions.append([self.current_theta, self.current_phi])

        self.spc.continuous_carve(self.current_theta, self.current_phi)

        self.reward = self.spc.gt_compare()

        pos = self.spc.radius*angle_to_position_continuous(
            self.current_theta, self.current_phi)
        self.pos = pos
        img = self.spc.img
        pil_img = Image.fromarray(img).resize((self.img_shape, self.img_shape))
        self.img = np.array(pil_img)
        img_gray = np.array(pil_img.convert('L'))
        self.img_gray = img_gray[...,None]
        self.penalty = np.linalg.norm(pos - self.last_k_pos[-3:])
        self.last_k_pos = np.concatenate((self.last_k_pos[3:], pos))

        #self.opt_dist = np.linalg.norm(self.pos - self.opt_pos, axis=1)

        self.total_reward += self.reward
        if self.reward > 0:
            self.reward -= self.k_d*self.penalty
        else:
            self.reward = -self.k_t

        canny = cv.Canny(img_gray, 40, 80)[...,None] * 1.
        self.canny = canny
                        
        self.last_k_img = np.concatenate((self.last_k_img[1:], self.canny[None]), axis=0) # shape (k,img_size,img_size,1)
        self.observation = self.last_k_img.astype('float32')

        #self.observation = self.last_k_pos.astype('float32')

        self.observation_shape = self.observation.shape

        if self.total_reward > self.max_reward:
            return dm_env.termination(reward=self.reward*1., observation=self.observation)

        return dm_env.transition(reward=self.reward*1., observation=self.observation)


    def segmentation_mask(self):

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
        return np.array(pil_img)


    def observation_spec(self) -> specs.BoundedArray:
        '''if (len(self.env_shape) == 3) & (self.apply_ae is None):
            return specs.BoundedArray(shape=self.env_shape, minimum=0, maximum=255, dtype=np.uint8)
        return specs.BoundedArray(shape=self.env_shape, minimum=-1, maximum=1, dtype=np.float32)'''
        pass

    def action_spec(self) -> specs.BoundedArray:
        '''return specs.BoundedArray(shape=(2,), minimum=-1, maximum=1, dtype=np.float32)'''
        pass

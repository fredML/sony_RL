import open3d as o3d
import utils as ut
import numpy as np
from skimage.morphology import binary_dilation
import proc3d
from vscan import virtual_scan
import json
import urllib
from scipy.spatial import cKDTree as KDTree
import collections
from acme import specs

TimeStep = collections.namedtuple("TimeStep", field_names=["step_type", "reward", "discount", "observation"])


class scanner_env():

    def __init__(self, params):
        self.gt = o3d.io.read_point_cloud(params["gt_path"])
        # calculate kdtree for gt once and for all
        self.gt_tree = KDTree(np.asarray(self.gt.points))
        self.vscan = virtual_scan(
            w=params['scanner']['w'], h=params['scanner']['h'], f=params['scanner']['f'])
        # ajouter paramètre de localisation
        self.vscan.load_im(params["plant_path"])
        self.N_theta = 2*np.pi/params["traj"]["N_theta"]
        self.N_phi = np.pi/params["traj"]["N_phi"]
        self.z0 = params["traj"]["z0"]
        self.R = params["traj"]["R"]
        self.n_dilation = params["sc"]["n_dilation"]
        self.voxel_size = params['sc']['voxel_size']
        self.origin, self.sc, self.bbox = ut.set_sc(self.vscan, self.voxel_size)
        self.intrinsics = ut.get_intrinsics(self.vscan)
        self.theta = self.phi = 0
        self.x = self.y = self.z = 0
        self.rx = self.ry = self.rz = 0
        self.count_steps = 0
        self.max_steps = 100

        self.obs_dim = params['train']['obs_dim']
        self.action_dim = params['train']['action_dim']
        self.discount = params['train']['discount']
        self.cd_min = 0.2

    def observation_spec(self) -> specs.BoundedArray:
        """Returns the observation spec."""
        return specs.BoundedArray(
            shape=self.obs_dim,
            name="pcd frames",
            minimum=0,
            maximum=255,
        )

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec."""
        return specs.DiscreteArray(
            dtype=int, num_values=np.prod(self.action_dim), name="action")

    def chamfer_d(self):
        vol = self.sc.values()
        vol = vol.reshape(self.sc.shape)
        pcd = proc3d.vol2pcd_exp(
            vol, self.origin, self.voxel_size, level_set_value=0)

        return ut.chamfer_d(self.gt_tree, np.asarray(
            self.gt.points), np.asarray(pcd.points))

    def get_reward(self):
        cd = self.chamfer_d()
        # delta_cd = self.cd/cd - 1 # do better than previous cd
        delta_cd = self.cd/cd - self.count_steps/self.max_steps*self.cd/self.cd_min
        self.cd = cd

        if cd < self.cd_min:
            done = True
            reward = 100
        else:
            done = False
            reward = delta_cd

        return reward, done

    def space_carve(self, im):
        mask = ut.get_mask(im)
        self.mask = mask
        rt = json.loads(urllib.request.urlopen(
            self.vscan.localhost + 'camera_extrinsics').read().decode('utf-8'))
        rot = sum(rt['R'], []) #flatten ?
        tvec = rt['T']
        if self.n_dilation:
            for k in range(self.n_dilation):
                mask = binary_dilation(mask)
        self.sc.process_view(self.intrinsics, rot, tvec, mask)

    def increase_phi(self):
        self.phi += self.N_phi
        self.phi = self.phi % (2*np.pi)
        self.update_pose()

    def decrease_phi(self):
        self.phi -= self.N_phi
        if self.phi < 0:
            self.phi += 2*np.pi
        self.update_pose()

    def increase_theta(self):
        self.theta += self.N_theta
        if np.abs(self.theta-np.pi) < 1e-4:
            self.theta = np.pi - self.N_theta
            self.phi = (self.phi + np.pi) % (2*np.pi)
        self.update_pose()

    def decrease_theta(self):
        self.theta -= self.N_theta
        if np.abs(self.theta) < 1e-4:
            self.theta = self.N_theta
            self.phi = (self.phi + np.pi) % (2*np.pi)
        self.update_pose()

    def change_pose(self, action):
        if action == 0:
            self.increase_theta()
        if action == 1:
            self.decrease_theta()
        if action == 2:
            self.increase_phi()
        if action == 3:
            self.decrease_phi()

    def update_pose(self):

        # theta and phi are spherical coordinate angles (here in multiples of N_phi rad)
        
        self.x = self.R * np.sin(self.theta) * np.cos(self.phi)
        self.y = self.R * np.sin(self.theta) * np.sin(self.phi)
        self.z = self.z0 + self.R * np.cos(self.theta)
        # where do we want the camera to face ? the origin ? first rz then rx (normally)
        #self.rx = 90 - self.theta*180/np.pi
        #self.rz = 90 + self.phi*180/np.pi

        self.rx = self.theta*180/np.pi
        self.rz = self.phi*180/np.pi + 90

    def reset(self, theta=None, phi=None):
        del(self.sc)
        self.origin, self.sc, self.bbox = ut.set_sc(self.vscan, self.voxel_size)
        self.count_steps = 0
        if theta is None:
            self.theta = np.random.choice(
                np.arange(self.N_theta, np.pi, self.N_theta))
        else:
            self.theta = theta 
        if phi is None:
            self.phi = np.random.choice(
                np.arange(self.N_phi, 2*np.pi+self.N_phi, self.N_phi))
        else:
            self.phi = phi

        self.update_pose()
        im = self.vscan.render(self.x, self.y, self.z,
                               self.rx, self.ry, self.rz)
        self.im = np.array(im)
        self.space_carve(im)
        #self.cd = self.chamfer_d()
        #fr = ut.get_frame(im)
        #self.state = np.array(
        #    [fr, fr, fr], dtype=np.float32).transpose(1, 2, 0)
        
        self.state = np.array(
            [self.im[...,0], self.im[...,0], self.im[...,0]]).transpose(1, 2, 0)

        return TimeStep(step_type=0, reward=None, discount=self.discount, observation=self.state)

    def step(self, action):
        self.count_steps += 1
        self.change_pose(action)
        im = self.vscan.render(self.x, self.y, self.z,
                               self.rx, self.ry, self.rz)
        self.im = np.array(im)
        self.space_carve(im)
        #fr = ut.get_frame(im)
        #self.state = jnp.array(
        #    [self.state[:, :, 1], self.state[:, :, 2], fr], dtype=jnp.float32).transpose(1, 2, 0)
        self.state = np.array(
            [self.state[:, :, 1], self.state[:, :, 2], self.im[...,0]]).transpose(1, 2, 0)
        #reward, done = self.get_reward()

        if self.count_steps > self.max_steps:
            done = True
        
        else:
            done = False

        return TimeStep(step_type=done, reward=0, discount=self.discount, observation=self.state)

    
    def step_continuous(self, action):
        self.count_steps += 1
        self.theta += action[0]
        self.theta = np.abs(self.theta % np.pi)
        self.phi += action[1]
        self.phi = self.phi % np.pi
        if self.phi < 0:
            self.phi += 2*np.pi
        im = self.vscan.render(self.x, self.y, self.z,
                               self.rx, self.ry, self.rz)
        self.space_carve(im)
        fr = ut.get_frame(im)
        self.state = np.array(
            [self.state[:, :, 1], self.state[:, :, 2], fr]).transpose(1, 2, 0)
        reward, done = self.get_reward()

        if self.count_steps > self.max_steps:
            done = True

        return TimeStep(step_type=done, reward=reward, discount=self.discount, observation=self.state)

    def step_angle(self, theta, phi):  # useful for predefined policies
        self.theta = theta
        self.phi = phi
        self.update_pose()

        im = self.vscan.render(self.x, self.y, self.z,
                               self.rx, self.ry, self.rz)
        self.im = im
        self.space_carve(im)

    def generate_img_masks(self, path, i_theta, i_phi, d_theta, d_phi):
        self.theta = i_theta*d_theta
        self.phi = i_phi*d_phi
        self.update_pose()

        im = self.vscan.render(self.x, self.y, self.z,
                               self.rx, self.ry, self.rz)
        mask = ut.get_mask(im)
        rt = urllib.request.urlopen(
            self.vscan.localhost + 'camera_extrinsics').read().decode('utf-8')
        rt = json.loads(rt)
        with open(path + f'/extrinsics/{i_theta:03d}_{i_phi:03d}.json', 'w') as f: 
            json.dump(rt,f)
        im.save(path + f'/imgs/{i_theta:03d}_{i_phi:03d}.png')
        np.save(path + f'/masks/{i_theta:03d}_{i_phi:03d}', mask)

    def camera_params(self):
        camera_model = json.loads(urllib.request.urlopen(
        self.vscan.localhost + 'camera_intrinsics').read().decode('utf-8'))
        bbox = json.loads(urllib.request.urlopen(self.vscan.localhost + 'bounding_box').read().decode('utf-8'))
        return camera_model, bbox

    def close(self) -> None:
        pass  # write any command to close server, etc.

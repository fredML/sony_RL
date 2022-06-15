import cv2
from scipy.ndimage import rotate
from cl import Backprojection
from proc3d import vol2pcd_exp
import numpy as np
from skimage.morphology import binary_dilation
import json
from PIL import Image, ImageOps
import glob
import os
from copy import deepcopy

class space_carving_rotation_2d():
    def __init__(self, model_path, phi_bias=0, total_phi_positions=180, cube_view='static', voxel_weights = None):
        # if dynamic, perspective of the camera seeing the cube changes according to position
        # if static, perspective is always the same
        self.cube_view = cube_view
        
        # phi position bias for simulating rotation of position of the object
        self.phi_bias = phi_bias
        
        # number of posible positions around the circle
        self.total_phi_positions = total_phi_positions
        
        # get all .png file names of images from folder path
        self.img_files = sorted (
            glob.glob(os.path.join(model_path, 'imgs', '*.png')) )
            
        # get all .png file names of image masks from folder path
        self.masks_files = sorted(
            glob.glob(os.path.join(model_path, 'masks', '*.png')))

        self.gt_files = sorted(
            glob.glob(os.path.join(model_path, 'ground_truth_volumes', '*.npy')))
        
        self.extrinsics = self.load_extrinsics(
            os.path.join(model_path, 'extrinsics'))
        
        self.bbox = json.load(
            open(os.path.join(model_path, 'bbox_general.json')))
   
        self.camera_model = json.load(
            open(os.path.join(model_path, 'camera_model.json')))
        self.intrinsics = self.camera_model['params'][0:4]

        params = json.load(open(os.path.join(model_path, 'params.json')))
        self.n_dilation = params["sc"]["n_dilation"]
        self.voxel_size = params['sc']['voxel_size']

        self.set_sc(self.bbox)
        self.vol_shape = self.sc.values().shape
        self.voxel_weights = voxel_weights

        ## reweighting of the voxels

        '''R = 0.3
        H = 8

        neigh_ijk = []
        self.index = np.ones(self.vol_shape)
        self.voxel_weights = np.zeros(self.vol_shape)

        if pos is not None:

            neigh_xyz = pos + np.array([[-h,r*np.cos(theta),r*np.sin(theta)] for theta in
                np.linspace(0,2*np.pi,10) for r in np.linspace(0,R,5) for h in np.linspace(0,H,10)])
            neigh_ijk.append((neigh_xyz - self.origin)/self.voxel_size)
            
            r = Rot.from_rotvec(np.pi/4 * np.array([1, 0, 2]))
            neigh_xyz_1 = r.apply(neigh_xyz)
            neigh_ijk.append((neigh_xyz_1 - self.origin)/self.voxel_size)

            neigh_xyz_2 = deepcopy(neigh_xyz_1)
            neigh_xyz_2[:,1] = -neigh_xyz_2[:,1]
            neigh_ijk.append((neigh_xyz_2 - self.origin)/self.voxel_size)
            neigh_ijk = np.concatenate(neigh_ijk)
            neigh_ijk = neigh_ijk.astype('int32')
            self.neigh_ijk = neigh_ijk

            for i in neigh_ijk:
                self.index[i[0],i[1],i[2]] = 0
                self.voxel_weights[i[0],i[1],i[2]] = 1

        self.voxel_weights = distance_transform_edt(self.index)
        self.voxel_weights = 1-self.voxel_weights/np.max(self.voxel_weights)'''

        # uses ground truth model 
        self.gt = np.load(self.gt_files[phi_bias])
        self.gt_solid_mask = np.where(self.gt == 1, True, False)
        self.gt_n_solid_voxels = np.sum(self.gt_solid_mask*self.voxel_weights)

    def reset(self):
        del(self.sc)
        self.set_sc(self.bbox)

    def load_extrinsics(self, path):
        ext = []
        ext_files = glob.glob(os.path.join(path, '*.json'))
        assert len(ext_files) != 0, "json list is empty."
        for i in sorted(ext_files):
            ext.append(json.load(open(i)))
        return ext

    def load_mask(self, idx):
        img = cv2.imread(self.masks_files[idx], cv2.IMREAD_GRAYSCALE)
        self.mask = img
        return img
    
    def get_pcd(self):
        return vol2pcd_exp(self.last_volume, self.origin, self.voxel_size, level_set_value=0)
    
    def get_image(self, phi, theta):
        biased_phi = self.calculate_phi_position(phi, -self.phi_bias)
        image_idx = (self.total_phi_positions * theta) + biased_phi
        
        img = Image.open(self.img_files[image_idx])
        cp = img.copy()
        cp = cp.resize((256,256))
        #img = ImageOps.grayscale(img)
        img.close()

        return cp
    
    def set_sc(self, bbox):
        x_min, x_max = bbox['x']
        y_min, y_max = bbox['y']
        z_min, z_max = bbox['z']

        nx = int((x_max - x_min) / self.voxel_size) + 1
        ny = int((y_max - y_min) / self.voxel_size) + 1
        nz = int((z_max - z_min) / self.voxel_size) + 1

        self.origin = np.array([x_min, y_min, z_min])
        self.sc = Backprojection(
            [nx, ny, nz], [x_min, y_min, z_min], self.voxel_size)
        self.last_volume = deepcopy(self.sc.values())

    def carve(self, phi, theta):
        '''space carve in position theta(rows),phi(cols)
        theta and phi are steps of the scanner, not angles'''
        # if using phi bias, use image of biased (shifted)phi with extrinsics
        #of current position to create a rotate model
        biased_phi = self.calculate_phi_position(phi, -self.phi_bias)
        image_idx = (self.total_phi_positions * theta) + biased_phi
        
        extrinsics_idx = (self.total_phi_positions * theta) + phi
        
        mask = self.load_mask(image_idx)
        self.space_carve(mask, self.extrinsics[extrinsics_idx])

        volume = self.sc.values()
        
        if self.cube_view == 'dynamic':
            # rotate cube according to current camera position
            #moves current position's perspective to cube position 0 so position 0
            #in cube always shows current position's perspective
            self.volume = rotate(self.volume,
                            angle=-phi*(360//self.total_phi_positions),reshape=False)

        self.carved_voxels = ((volume == -1) & (self.last_volume != -1))*self.voxel_weights
        self.last_volume = deepcopy(volume)

    def space_carve(self, mask, rt):
        '''do space carving on mask with preset parameters'''
        # mask = im.copy() #mask = get_mask(im)
        rot = sum(rt['R'], [])
        tvec = rt['T']
        if self.n_dilation:
            for _ in range(self.n_dilation):
                mask = binary_dilation(mask)
        self.sc.process_view(self.intrinsics, rot, tvec, mask)

    def gt_compare(self):
        '''compares current volume with ground truth (voxelwise) and returns percentage'''
        #comp = (self.gt == self.last_volume)*self.voxel_weights
        #eq_count = comp.sum()
        eq_count = self.carved_voxels.sum()
        perc_sim = eq_count/(self.voxel_weights.sum())
        return perc_sim

    def gt_compare_solid(self):
        ''' returns intersection of solid voxels (with 1;s) between ground truth and test_vol.
        It is the ratio between the intersecting solid voxels of current volume and groundtruth, 
        and the total numer of solid voxels found in both volumes'''
        
        # gets solid voxels array in current volume
        vol_solid_mask = np.where(self.last_volume == 1, True, False)
        
        #gets the total number of solid voxels in current volume
        vol_n_solid_voxels = np.sum(vol_solid_mask*self.voxel_weights)
        
        #gets number of solid voxels that intersect current vol and GroundTruth
        intersection = (self.gt_solid_mask & vol_solid_mask)*self.voxel_weights
        n_intersection = np.sum(intersection)
        
        ratio = n_intersection / (self.gt_n_solid_voxels + vol_n_solid_voxels - n_intersection)
        return ratio

    def calculate_phi_position(self, init_state, steps):
        '''calculates phi position (in steps) from current phi plus n steps'''
        n_pos = init_state + steps
        if n_pos > (self.total_phi_positions-1):
            n_pos -= self.total_phi_positions
        elif n_pos < 0:
            n_pos += self.total_phi_positions
        return n_pos

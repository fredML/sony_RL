import cv2
import numpy as np
from skimage.morphology import binary_dilation
from skimage.transform import resize
import json
from PIL import Image, ImageOps
import glob
import os
from copy import deepcopy
from utils import angle_to_position_continuous, get_mask_black, set_sc, make_extrinsics

path = os.getcwd()
os.chdir('/mnt/diskSustainability/frederic/rlviewer')
os.environ['DISPLAY'] = ':1'
import rlviewer
os.chdir(path)

class space_carving_rotation_2d():
    def __init__(self, model_path, total_phi_positions=None, voxel_weights=None, continuous=True):

        self.camera_model = json.load(
                open(os.path.join(model_path, 'camera_model.json')))

        self.intrinsics = self.camera_model['params'][0:4]
        
        self.bbox = json.load(
            open(os.path.join(model_path, 'bbox_general.json')))

        params = json.load(open(os.path.join(model_path, 'params.json')))

        self.n_dilation = params["sc"]["n_dilation"]
        self.voxel_size = params['sc']['voxel_size']
        self.radius = params['traj']['R']

        self.origin, self.sc, self.last_volume = set_sc(self.bbox, self.voxel_size)
        self.vol_shape = self.sc.values().shape

        if voxel_weights is not None:
            self.voxel_weights = voxel_weights
        else:
            self.voxel_weights = np.ones(self.vol_shape)

        self.continuous = continuous

        if self.continuous:
            rlviewer.load(os.path.join(model_path,'sphere.obj')) 
            rlviewer.set_light(0, 120, 0, 0, 5000) 
            rlviewer.set_light(1, -120, 0, 0, 5000) 
            rlviewer.set_light(2, 0, 0, 120, 5000) 
            rlviewer.set_light(3, 0, 0, -120, 5000) 
        
        else:
            self.total_phi_positions = total_phi_positions

            self.img_files = sorted (
                glob.glob(os.path.join(model_path, 'imgs', '*.png')) )
        
            self.masks_files = sorted(
                glob.glob(os.path.join(model_path, 'masks', '*.png')))

            self.gt_files = sorted(
                glob.glob(os.path.join(model_path, 'ground_truth_volumes', '*.npy')))
            
            self.extrinsics = self.load_extrinsics(
                os.path.join(model_path, 'extrinsics'))

            self.gt = np.load(self.gt_files[0])
            self.gt_solid_mask = np.where(self.gt == 1, True, False)
            self.gt_n_solid_voxels = np.sum(self.gt_solid_mask*self.voxel_weights)

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

    def reset(self):
        del(self.sc)
        self.origin, self.sc, self.last_volume = set_sc(self.bbox, self.voxel_size)

    def load_extrinsics(self, path):
        ext = []
        ext_files = glob.glob(os.path.join(path, '*.json'))
        assert len(ext_files) != 0, "json list is empty."
        for i in sorted(ext_files):
            ext.append(json.load(open(i)))
        return ext

    def load_mask(self, idx):
        mask = cv2.imread(self.masks_files[idx], cv2.IMREAD_GRAYSCALE)
        self.mask = mask
        return mask
    
    def get_image(self, theta, phi):
        image_idx = (self.total_phi_positions * theta) + phi
        img = Image.open(self.img_files[image_idx])
        cp = img.copy()
        cp = cp.resize((256,256))
        #img = ImageOps.grayscale(img)
        img.close()

        return cp

    def carve(self, theta, phi):
        '''space carve in position theta(rows),phi(cols)
        theta and phi are steps of the scanner, not angles'''
   
        idx = (self.total_phi_positions * theta) + phi
        mask = self.load_mask(idx)
        self.space_carve(mask, self.extrinsics[idx])

        volume = self.sc.values()
        self.carved_voxels = ((volume == -1) & (self.last_volume != -1))*self.voxel_weights
        self.last_volume = deepcopy(volume)

    def continuous_carve(self, theta, phi):
        assert self.continuous, "continuous settings only"
        img = rlviewer.grab(self.radius, theta-np.pi/2, phi) 
        self.img = img
        mask = get_mask_black(img)
        self.mask = mask
        extrinsics = make_extrinsics(self.radius, angle_to_position_continuous(theta, phi))
        self.extrinsics = extrinsics
        self.space_carve(mask, self.extrinsics)

        volume = self.sc.values()
        self.carved_voxels = ((volume == -1) & (self.last_volume != -1))*self.voxel_weights
        self.last_volume = deepcopy(volume)
        
    def space_carve(self, mask, rt):
        '''do space carving on mask with preset parameters'''
        rot = sum(rt['R'], [])
        tvec = rt['T']
        '''if self.n_dilation:
            for _ in range(self.n_dilation):
                mask = binary_dilation(mask)'''
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

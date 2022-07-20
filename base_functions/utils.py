from scipy.spatial import cKDTree as KDTree
import numpy as np
import cv2
import json
import urllib
import cl
from copy import deepcopy

def chamfer_d(tree1, pc_1, pc_2):
    ds, _ = tree1.query(pc_2, workers=-1)
    d_21 = np.mean(ds)

    tree2 = KDTree(pc_2)
    ds, _ = tree2.query(pc_1, workers=-1)
    d_12 = np.mean(ds)
    return d_21 + d_12

def get_mask(img):
    res = np.array(img)[:, :, :3]
    mask1 = np.all((res == [70, 70, 70]), axis=-1)
    mask2 = np.all((res == [71, 71, 71]), axis=-1)
    mask3 = np.all((res == [72, 72, 72]), axis=-1)
    mask12 = mask1 | mask2
    mask_inv = mask12 | mask3
    mask = 1 - mask_inv
    return 255*mask

def get_mask_black(img):
    a = img[...,0]==0
    b = img[...,1]==0
    c = img[...,2]==0

    mask = 1 - (a&b&c)*1
    return 255*mask

def get_frame(img):
    img = np.array(img)[:, :, :3]
    mask1 = np.all((img == [70, 70, 70]), axis=-1)
    mask2 = np.all((img == [71, 71, 71]), axis=-1)
    mask3 = np.all((img == [72, 72, 72]), axis=-1)
    mask12 = mask1 | mask2
    mask = mask12 | mask3
    res = cv2.bitwise_and(img, img, mask=(255-255*mask).astype(np.uint8))
    res_bg = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(res_bg, (84, 84))
    return small

def set_sc(bbox,voxel_size):

  x_min, x_max = bbox['x']
  y_min, y_max = bbox['y']
  z_min, z_max = bbox['z']

  nx = int((x_max - x_min) / voxel_size) + 1
  ny = int((y_max - y_min) / voxel_size) + 1
  nz = int((z_max - z_min) / voxel_size) + 1

  origin = np.array([x_min, y_min, z_min])
  sc = cl.Backprojection([nx, ny, nz], [x_min, y_min, z_min], voxel_size)
  last_volume = deepcopy(sc.values())
  return origin, sc, last_volume

def get_intrinsics(scanner):
    url_part = 'camera_intrinsics'
    camera_model = json.loads(urllib.request.urlopen(
        scanner.localhost + url_part).read().decode('utf-8'))
    return camera_model['params'][0:4]

'''def make_extrinsics(pos, up): #we want to face the X-axis (forward axis) using rotation R
    norm_pos = pos/np.linalg.norm(pos)
    if all(norm_pos==up): #if on up axis, need to rotate manually by -90deg around Y axis
        R = np.array([[0,0,-1],[0,1,0],[1,0,0]])
    else:
        R = np.zeros((3,3))
        R[2] = norm_pos
        R[0] = np.cross(up, norm_pos)
        R[1] = np.cross(R[2],R[0])

    T = np.dot(R,-pos)
    return {'R':R.tolist(), 'T':T.tolist()}'''

def make_extrinsics(radius, pos, up):
    #pos = pos/np.linalg.norm(pos)
    n = len(pos)
    up = np.array([0,0,1])
    R = np.zeros((n,n))
    R[2] = pos
    R[0] = np.cross(up,pos)
    R[0] = R[0]/np.linalg.norm(R[0])
    R[1] = np.cross(R[2],R[0])
    R[1] = R[1]/np.linalg.norm(R[1])

    T = np.array([0, 0, radius])

    return {'R':R, 'T':T}

def angle_to_position(theta, phi):
    return np.array([np.cos(2*phi*np.pi/180)*np.sin((theta+1)*np.pi/8),
                    np.sin(2*phi*np.pi/180)*np.sin((theta+1)*np.pi/8),
                    np.cos((theta+1)*np.pi/8)])

def angle_to_position_continuous(theta, phi):
    return np.array([np.cos(phi)*np.sin(theta),
                     np.sin(phi)*np.sin(theta),
                     np.cos(theta)])
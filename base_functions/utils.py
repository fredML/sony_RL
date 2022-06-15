from scipy.spatial import cKDTree as KDTree
import numpy as np
import cv2
import chex 
import jax
import json
import urllib
import cl

def chamfer_d(tree1, pc_1, pc_2):
    #tree = KDtree(pc_1)
    #ds, _ = tree.query(pc_2)
    ds, _ = tree1.query(pc_2, workers=-1)
    d_21 = np.mean(ds)

    tree2 = KDTree(pc_2)
    #ds, _ = tree.query(pc_1)
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

def set_sc(scanner,voxel_size):

  bbox = json.loads(urllib.request.urlopen(scanner.localhost + 'bounding_box').read().decode('utf-8'))
  x_min, x_max = bbox['x']
  y_min, y_max = bbox['y']
  z_min, z_max = bbox['z']

  nx = int((x_max - x_min) / voxel_size) + 1
  ny = int((y_max - y_min) / voxel_size) + 1
  nz = int((z_max - z_min) / voxel_size) + 1

  origin = np.array([x_min, y_min, z_min])
  return origin, cl.Backprojection([nx, ny, nz], [x_min, y_min, z_min], voxel_size), bbox

def get_intrinsics(scanner):
    url_part = 'camera_intrinsics'
    camera_model = json.loads(urllib.request.urlopen(
        scanner.localhost + url_part).read().decode('utf-8'))
    return camera_model['params'][0:4]

def val(features,i):
  return features[i] 

def batched_indexing(features: chex.Array,idxs: chex.Array) -> chex.Array:
  return jax.vmap(val)(features,idxs)

def run_test(env, network_apply, online_params):
    chamfer_d = []
    positions = []

    ts = env.reset(theta=np.pi/2,phi=0)
    chamfer_d.append(env.chamfer_d())
    positions.append([env.x, env.y, env.z])

    for k in range (19):
        q = network_apply(online_params, ts.observation[None])[0]
        a = np.argmax(q)
        ts = env.step(a)
        chamfer_d.append(env.chamfer_d())
        positions.append([env.x, env.y, env.z])

    return chamfer_d, positions
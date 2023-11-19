# %%
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import collections

import h5py
import matplotlib.pyplot as plt
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robosuite.utils.transform_utils as T
import yaml
from moviepy.editor import ImageSequenceClip
from tqdm import trange

from diffusion_policy.common.camera_utils import CameraMover


def animate(imgs, filename='animation.mp4', _return=True, fps=10):
    if isinstance(imgs, dict):
        imgs = imgs['image']
    imgs = ImageSequenceClip(imgs, fps=fps)
    imgs.write_videofile(filename, fps=fps, logger=None)
    if _return:
        from IPython.display import Video
        return Video(filename, embed=False)
    

# dataset_path = 'data/robomimic/datasets/tool_hang/ph/image.hdf5'
# camera_name = 'sideview'

dataset_path = 'data/robomimic/datasets/can/ph/image.hdf5'
camera_name = 'agentview'

f = h5py.File(dataset_path, 'r')
states = f['data']['demo_0']['states']
actions = f['data']['demo_0']['actions']

# images = [x for x in f['data']['demo_0']['obs']['sideview_image']]
# animate(images)


shape_meta_yaml = """
obs:
    sideview_image:
        shape: [3, 84, 84]
        type: rgb
action: 
    shape: [10]
"""
shape_meta = yaml.load(shape_meta_yaml, Loader=yaml.Loader)
modality_mapping = collections.defaultdict(list)
for key, attr in shape_meta['obs'].items():
    modality_mapping[attr.get('type', 'low_dim')].append(key)
ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
# env_meta['env_kwargs']['camera_depths'] = True

# %%
env = EnvUtils.create_env_from_metadata(
    env_meta=env_meta,
    render=False, 
    render_offscreen=True,
    use_image_obs=True, 
)

camera_mover = CameraMover(
    env=env.env,
    camera=camera_name
)

env.reset_to({"states" : states[0]})
traj_len = states.shape[0]
# traj_len = 200

imgs = list()
for t in trange(traj_len - 1):
    action = actions[t].copy()
    # action[:3] = np.array([0, -0.2, 0])
    # action[:6] = np.array([np.sin(t/100.0), -np.cos(t/100.0), 0, 0, 0, 0]) * 0.2
    v_pos = action[:3]
    
    # obs, _, _, _ = env.step(action)
    obs = env.reset_to({"states" : states[t+1]})
    
    img = obs[f'{camera_name}_image']
    img = (np.moveaxis(img, 0, -1) * 255).astype(np.uint8)
    imgs.append(img)

    cur_camera_pos, cur_camera_quat = camera_mover.get_camera_pose()
    camera_rot = T.quat2mat(cur_camera_quat)
    dot_product= list()
    angle = 3 # degree
    for sgn in [1, 0, -1]:
        rad = sgn * np.pi * angle / 180.0
        R = T.rotation_matrix(rad, [0, 0, 1], point=None)
        camera_pose = np.zeros((4, 4))
        camera_pose[:3, :3] = camera_rot
        camera_pose[:3, 3] = cur_camera_pos
        camera_pose = R @ camera_pose
        dot_product.append(camera_pose[:3, 3].dot(v_pos))
    max_idx = np.argmin(np.abs(dot_product))
    if max_idx == 0:
        camera_mover.rotate_camera_world(None, [0, 0, 1], angle)
    elif max_idx == 2:
        camera_mover.rotate_camera_world(None, [0, 0, 1], -angle)
    # print(t, dot_product, v_pos)

animate(imgs, 'animation_new.mp4')
# %%

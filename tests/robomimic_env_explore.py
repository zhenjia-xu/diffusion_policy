# %%
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import collections

import matplotlib.pyplot as plt
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robosuite.utils.transform_utils as T
import yaml
from robomimic.algo import algo_factory
from robomimic.config import config_factory
from robomimic.utils.dataset import SequenceDataset

from diffusion_policy.common.camera_utils import CameraMover

dataset_path = 'data/robomimic/datasets/tool_hang/ph/image.hdf5'

# default BC config
config = config_factory(algo_name="bc")

# read config to set up metadata for observation modalities (e.g. detecting rgb observations)
# must ran before create dataset
# ObsUtils.initialize_obs_utils_with_config(config)

shape_meta_yaml = """
obs:
    agentview_image:
        shape: [3, 84, 84]
        type: rgb
    sideview_image:
        shape: [3, 84, 84]
        type: rgb
    robot0_eye_in_hand_image:
        shape: [3, 84, 84]
        type: rgb
    robot0_eef_pos:
        shape: [3]
        # type default: low_dim
    robot0_eef_quat:
        shape: [4]
    robot0_gripper_qpos:
        shape: [2]
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
env = EnvUtils.create_env_from_metadata(
    env_meta=env_meta,
    render=False, 
    render_offscreen=True,
    use_image_obs=True, 
)

obs = env.reset()
print(obs.keys())
# plt.imshow(obs['agentview_image'].transpose([1, 2, 0]))

#%%
camera_mover = CameraMover(
    env=env.env,
    # camera='agentview'
    camera='sideview'
)
camera_mover.rotate_camera_world(point=[0, 0, 0], axis=[0.0, 0.0, 1.0], angle=90, rel_init=True)
# plt.imshow(env.get_observation()['agentview_image'].transpose([1, 2, 0]))
plt.imshow(env.get_observation()['sideview_image'].transpose([1, 2, 0]))


obs, reward, done, info = env.step(np.ones(7))

camera_pos, camera_quat_wxyz = camera_mover.get_camera_pose()
camera_quat_xyzw = T.convert_quat(camera_quat_wxyz, to="xyzw")
camera_rot = T.quat2mat(camera_quat_xyzw)

camera_quat_inv_xyzw = T.quat_inverse(camera_quat_xyzw)
camera_rot_inv = T.quat2mat(camera_quat_inv_xyzw)

dist = np.linalg.norm(camera_pos)


# #%%
# for i in range(10):
#     obs, reward, done, info = env.step(np.ones(7))
#     robot = env.env.robots[0]
#     controller = robot.controller
#     gripper = robot.gripper
#     ee_ori_mat = controller.ee_ori_mat
#     ee_pos = controller.ee_pos
#     print(ee_pos)
# # %%


# env = EnvUtils.create_env_for_data_processing(
# # https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/scripts/dataset_states_to_obs.py
# %%

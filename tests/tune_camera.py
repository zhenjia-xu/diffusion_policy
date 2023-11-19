#%%
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import robomimic
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.test_utils as TestUtils
import robomimic.utils.torch_utils as TorchUtils

from diffusion_policy.common.camera_utils import CameraMover

#%%

shape_meta_yaml = """
obs:
    agentview_image:
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

dataset_path = 'data/robomimic/datasets/transport/ph/image.hdf5'
env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
# env_meta['env_kwargs']['camera_depths'] = True
env = EnvUtils.create_env_from_metadata(
    env_meta=env_meta,
    render=False, 
    render_offscreen=True,
    use_image_obs=True, 
)
print(env.reset().keys())
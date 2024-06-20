import gym
from nvf.active_mapping.active_mapping import ActiveMapper
from nvf.active_mapping.mapping_utils import to_transform
# from gym import spaces
import gtsam
import numpy as np

from nvf.env.utils import pose_point_to, get_conf, get_transforms, get_images, rgb_to_rgba, GIFSaver, save_img, pose2tensor

import torch
from PIL import Image
from PIL import ImageDraw

from nvf.env.Scene import *


class Enviroment(gym.Env):
    # init_images = 10
    # horizon = 20
    """Custom Environment that follows gym interface"""
    def __init__(self, cfg,):
        super(Enviroment, self).__init__()

        self.horizon = cfg.horizon
        self.max_steps = cfg.horizon
        
        self.pose_history = []
        self.obs_history = []

        self.cfg = cfg

        # init nerf pipeline
        # self.pipeline = ActiveMapper()
        # config_path = self.pipeline.initialize_config(config_home = "/home/jdill/nerfstudio/nerfstudio/scripts/pipeline_testing/temp", dataset_path = "/home/jdill/nerfstudio/nerfstudio/scripts/pipeline_testing/dataset")
        # self.pipeline.clear_dataset()


        # init blender env
        # cfg = get_conf()
        # self.scene = HubbleScene(cfg)
        self.scene = eval(self.cfg.scene)(cfg)

        

        # Initialize state
        self.reset()

    def step(self, action):
        
        position = self.state

        # new_poses = []
        new_images = []
        for pose in action:
            img = self.scene.render_pose(pose)

            # self.scene.set_camera_pose(pose)
            # result = self.scene.render()
            # # img = result['mask']*result['image']
            # img = rgb_to_rgba(result['image']*result['mask'], result['mask'])
            # input(img.shape)
            # new_poses.append(pose)
            new_images.append(img)
        

        
        # self.pipeline.add_image(images=new_images, poses=action)

        self.pose_history += action
        self.obs_history += new_images

        done = False
        if self.steps >= self.max_steps:
            done = True

        reward = 0. # TODO

        self.state = action[-1]
        self.steps += 1


        return new_images, reward, done, {}

    def reset(self):
        # transforms = get_transforms(idx_start=0, idx_end=100)

        # np_images = []
        # for pose in transforms:
        #     self.scene.set_camera_pose(pose)
        #     result = self.scene.render()
        #     img = result['mask']*result['image']
        #     img = rgb_to_rgba(img)
        #     # input(img.shape)
        #     # new_poses.append(pose)
        #     np_images.append(img)
        
        np_images, transforms = self.get_images(mode='init')

        self.pose_history = transforms
        self.obs_history = np_images
        self.state = np.array(self.pose_history[-1])  # Reset position to the center

        # self.pipeline.add_image(images=np_images, poses=transforms)
        
        self.steps = 0
        return np_images  # reward, done, info can't be included
    
    def get_images(self, mode, return_quat=True):
        file = f'data/{self.cfg.scene.name}/{mode}/transforms.json'
        if not os.path.exists(file) or (hasattr(self.cfg, f'gen_{mode}') and getattr(self.cfg, f'gen_{mode}')):
            return self.gen_data(mode, return_quat=return_quat)
        else:
            img, transforms = self.scene.load_data(file=file, return_quat=return_quat)
            if mode == 'init':
                if len(img) != self.cfg.n_init_views:
                    return self.gen_data(mode, return_quat=return_quat)
            return img, transforms
        
    
    def gen_data(self, mode, return_quat=False):
        file = f'data/{self.cfg.scene.name}/{mode}/transforms.json'
        print(f'Generating data for {mode}')
        poses = self.scene.gen_data_fn[mode]()

        images = self.scene.render_poses(poses)
        
        if getattr(self.cfg, f'save_data'):
            print(f'saving data to {file}')
            self.scene.save_data(file, poses, images)
        
        if return_quat:
            poses = [pose2tensor(pose) for pose in poses]
        else:
            poses = [torch.FloatTensor(pose) for pose in poses]
        return images, poses
        

def test_Enviroment():
    env = Enviroment()

    agent = BaseAgent()
    
    action = agent.act(env.obs_history, env.pose_history)
    # input(action)
    print(action)

    gif = GIFSaver()
    for i in range(20):
        obs, _, _, _ = env.step(action)
        save_img(obs[0], f'results/test_sim_2_{i}.png')
        print(type(obs[0]), obs[0].shape)
        gif.add(obs[0], fn=lambda img: ImageDraw.Draw(img).text((3, 3), f"T={i}", fill=(255, 255, 255)))
        action = agent.act(obs, action)
        # gif.save('results/test_sim1.gif')

    print(action)
    gif.save('results/test_sim2.gif')
    input()
    pass

if __name__ == "__main__":
    from nvf.active_mapping.agents.BaseAgent import *
    test_Enviroment()
    pass

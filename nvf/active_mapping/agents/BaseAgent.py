
import gtsam
import numpy as np
import torch

from nvf.active_mapping.agents.Sampler import *
from nvf.active_mapping.active_mapping import ActiveMapper
from nvf.active_mapping.mapping_utils import to_transform
from nvf.env.utils import pose2tensor, tensor2pose

from nvf.env.utils import empty_cache
from nvf.metric.mesh_metrics import VisibilityMeshMetric, FaceIndexShader

from dataclasses import dataclass, field

from nvf.env.Scene import *
from nvf.env.utils import get_conf
from nvf.env.utils import get_images

# from eval import set_env

from torch.cuda.amp import GradScaler, autocast
import time

class BaseAgent():
    use_ckpts:bool = False

    def __init__(self, config):
        # super().__init__()
        self.config = config
        # utils.set_seed(0)

        # init nerf pipeline
        self.pipeline = ActiveMapper()
        self.pipeline.fov = config.env.fov
        self.pipeline.train_img_size = config.env.resolution
        config_path = self.pipeline.initialize_config(config_home = "cfg/", dataset_path = "outputs/pipeline/dataset", model=config.model)

        # self.pipeline.clear_dataset()
        # self.pipeline.toggle_config_model_checkpoint(True)
        # self.pipeline.toggle_config_model_checkpoint(True)
        # self.sampler = AABBSampler()
        self.sampler = eval(self.config.sampler)(config)
        self.sampler.pipeline = self.pipeline
        self.n_sample = self.config.n_sample
        

        self.step = 0

        self.plan_hist = []
        self.obs_hist = []
        self.pose_hist = []

        self.time_record = {'plan_time':[], 'train_time':[]}

    def process_obs(self, obs, prev_poses):
        self.obs_hist += obs
        self.pose_hist += prev_poses
        if self.step ==0:
            add_image_option = None
            self.start_step = len(obs)
        else:
            add_image_option = None#'reinit'
        if self.step == self.config.horizon-1:
            print('agent last step')
        
        self.pipeline.add_image(images=obs, poses=prev_poses, model_option=add_image_option)
        self.current_pose = tensor2pose(prev_poses[-1])
    
    def get_reward(self, poses):
        enren  = self.pipeline.trainer.pipeline.model.renderer_entropy
        d0 = '' if not hasattr(enren, 'depth_threshold') else f'd0: {enren.depth_threshold}'
        print('Entropy Type:',type(enren), d0)
        
        plan_result = {"pose":poses.detach().cpu().numpy()}

        with torch.no_grad():
            cost = self.pipeline.get_cost(poses=poses[:,None,:], return_image=True)
            # plan_result["entropy"]= cost.detach().cpu().numpy()
            cost = cost.mean(dim=(-1, -2))
            plan_result["entropy"]= cost.detach().cpu().numpy()
            # print(cost.shape)
        self.plan_hist.append(plan_result)

        return cost
       

    def act(self, obs,prev_poses):
        t0 = time.time()
        empty_cache()
        print('Start Training NeRF')
        self.process_obs(obs, prev_poses)

        t1 = time.time()
        print('Start Planning')
        empty_cache()

        poses = self.sampler(self.n_sample, pose=self.current_pose)
        
        cost = self.get_reward(poses)

        best_idx = cost.argmax().item()
        best_cost = cost[best_idx].item()

        best_pose = poses[best_idx, ...]
        empty_cache()

        self.step +=1

        t2 = time.time()
        self.time_record['train_time'].append(t1-t0)
        self.time_record['plan_time'].append(t2-t1)
        return [best_pose]

class RandomAgent(BaseAgent):
    train_each_iter = True

    def __init__(self, config):
        super().__init__(config)
        self.sampler.pipeline = None
    
    def process_obs(self, obs, prev_poses):
        self.obs_hist += obs
        self.pose_hist += prev_poses
        if self.train_each_iter:
            if self.step ==0:
                add_image_option = None
            else:
                add_image_option = 'reinit'
            self.pipeline.add_image(images=obs, poses=prev_poses, model_option=add_image_option)  
                  
        elif self.step == self.config.horizon-1:
            self.pipeline.add_image(images=self.obs_hist, poses=self.pose_hist)
        
        self.current_pose = tensor2pose(prev_poses[-1])

    def get_reward(self, poses):
        # breakpoint()
        return torch.rand(poses.shape[0], 1, device=poses.device)
    
class OptAgent(BaseAgent):
    use_ckpts:bool = False
    
    def get_reward(self, poses):
        plan_result = {"pose":poses.detach().cpu().numpy()}

        with torch.no_grad():
            cost = self.pipeline.get_cost(poses=poses[:,None,:], return_image=True)
            plan_result["entropy"]= cost.detach().cpu().numpy()
            cost = cost.mean(dim=(-1, -2))
        
        self.plan_hist.append(plan_result)

        return cost
       
    def act(self, obs,prev_poses):
        t0 = time.time()
        empty_cache()
        print('Start Training NeRF')
        self.process_obs(obs, prev_poses)
        t1 = time.time()
        print('Start Planning')
        empty_cache()        

        device = self.pipeline.trainer.device
        cpu_or_cuda_str: str = self.pipeline.trainer.device.split(":")[0]
        mixed_precision = self.pipeline.trainer.mixed_precision
        aabb = self.config.camera_aabb.to(device)

        # Specify top-k value
        k=self.config.n_opt

        poses_ = self.sampler(self.n_sample, pose=self.current_pose)
        cost = self.pipeline.get_cost(poses=poses_[:,None,:], return_image=True)
        cost = cost.mean(dim=(-1, -2))
        # Get top-k costs
        k_costs,k_idxs = torch.topk(cost.squeeze(1),k)
        topk_poses = poses_[k_idxs,...]
        topk_cost = cost[k_idxs,...]

        poses = torch.tensor(poses_[k_idxs,...], device=device, requires_grad=True)
        # poses.requires_grad = True
        scaler = GradScaler()
        optimizer = torch.optim.Adam([poses] ,lr=self.config.opt_lr)
        print(f"Top {k} Poses Pre Optimization:{poses_[k_idxs,...]}")
        print(f"Cost Pre Optimization:{cost[k_idxs,...].view(-1)}")
        for iter in range(self.config.opt_iter):
            optimizer.zero_grad()
            with torch.autocast(device_type=cpu_or_cuda_str, enabled=mixed_precision):
                cost = -self.pipeline.get_cost(poses[:,None,:], return_image=False)
                

            if cost.requires_grad:
                try: 
                    scaler.scale(cost.sum()).backward(retain_graph=True)
                    scaler.step(optimizer)
                    scaler.update()
                except Exception as error:
                    # likely cased by no samples within instant-ngp
                    print('error occurs in pose optimization!!!')
                    print(error)
            
            # Normalize quaternion and clip pose according to camera using aabb 
            with torch.no_grad():
                for i in range(0,k):
                    # print("Pose Pre Quaternion Normalization",poses[i])
                    quat_norm = torch.norm(poses[i][0:4], p=2).clone()
                    quat = (poses[i][0:4]/quat_norm)
                    poses[i][0:4]=quat.clone()

                    
                    poses[i][4] = torch.clip(poses[i][4],aabb[0][0],aabb[1][0])
                    poses[i][5] = torch.clip(poses[i][5],aabb[0][1],aabb[1][1])
                    poses[i][6] = torch.clip(poses[i][6],aabb[0][2],aabb[1][2])
                    # print("Pose Post Quaternion Normalization",poses[i])

        best_poses = poses.detach().cpu().clone()

        mask = ~torch.isnan(best_poses).any(dim=1)
        best_poses = best_poses[mask]

        plan_result = {"pose":poses.detach().cpu().numpy()}
        
        use_init_poses = True # use init (topk) pose for selection
        if use_init_poses:
            with torch.no_grad():
                best_poses = torch.cat([best_poses, topk_poses], dim=0)
                cost = self.pipeline.get_cost(poses=best_poses[:,None,:], return_image=True)
        else:
            if best_poses.shape[0] == 0:
                print('No valid poses found, using topk poses')
                best_poses = topk_poses
                cost = topk_cost.unsqueeze(-1).unsqueeze(-1)
            else:
                cost = self.pipeline.get_cost(poses=best_poses[:,None,:], return_image=True)

        
        plan_result["entropy"]= cost.detach().cpu().numpy()
        cost = cost.mean(dim=(-1, -2))
        self.plan_hist.append(plan_result)
        
        best_pose = best_poses[cost.argmax().item(),...]
        print(f"Pose Post Optimization:{best_pose}")
        print(f"Cost Post Optimization:{cost.view(-1)}")
        self.step +=1
        del poses
        del poses_
        del best_poses
        del cost

        t2 = time.time()
        self.time_record['train_time'].append(t1-t0)
        self.time_record['plan_time'].append(t2-t1)
        return [best_pose]

if __name__ == "__main__":
    BaseAgent()

    
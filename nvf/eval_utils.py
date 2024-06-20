import torch
import pickle as pkl
from enum import Enum
import pandas as pd

class StrEnum(str, Enum):
    """
    Enum where members are also (and must be) strings
    """
    def __repr__(self):
        return f'{self.value}'
    def __str__(self):
        return f'{self.value}'

def compute_entropy_corr(gt_img, pred_img, entropy):
    # err_gt = torch.sum((gt_img-pred_img)**2, dim=-3)**0.5
    err_gt = torch.sum(torch.abs(gt_img-pred_img), dim=-3)
    # corr = torch.corrcoef(torch.stack([err_gt.view(-1).cuda(), err_pred.view(-1).cuda()]))
    corr = torch.corrcoef(torch.stack([err_gt.sum(axis=(-1,-2)).cuda(), entropy.sum(axis=(-1,-2)).cuda()]))
    return corr[0,1].item()

def save_result(file, tracker, agent):
    tracker.metric_hist.update(agent.time_record)
    result = {
            'metric':tracker.metric_hist,
            'trajectory':tracker.trajectory,
            'trajectory_start_idx ':tracker.trajectory_start_idx,
            'pred_img_hist':tracker.pred_img_hist,
            'eval_entropy_hist':tracker.entropy_hist,
            'ref_pose':tracker.ref_poses.detach().cpu().numpy(),
            'ref_img':tracker.ref_images.detach().cpu().numpy(),
            'plan_hist': agent.plan_hist,
            'obs_hist':agent.obs_hist,
            'pose_hist':agent.pose_hist,
            # 'start_step':agent.start_step,
            
        }
    with open(file, 'wb') as f:
        pkl.dump(result, f)
    
    return result

def save_dict_to_excel(file, metric):
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(metric)

    # Save the DataFrame to an Excel file
    df.to_excel(file, index=True)

def get_corr(tracker, pipeline, ks=32, return_images=False):
    down_sampler = torch.nn.AvgPool2d(kernel_size=ks, stride=ks)
    pred_img = tracker.visualize(pipeline)
    pred_img = pred_img.to(pipeline.trainer.device)
    gt_img = tracker.ref_images.to(pred_img)

    err_gt = down_sampler(torch.sum(torch.abs(gt_img-pred_img), dim=-3))
    # err_gt = down_sampler(torch.sqrt(torch.sum((gt_img-pred_img)**2, dim=-3)))
    # print(err_gt.shape)
    # err_gt = err_gt.view(-1)
    # err_gt = (err_gt - err_gt.mean())/err_gt.std()
    # output = output.to(pipeline.trainer.device)
    # tracker.ref_images = tracker.ref_images.to(output)
    # breakpoint()
    # import matplotlib.pyplot as plt

    err_pred = down_sampler(tracker.get_entropy(pipeline))
    # err_pred = err_pred.view(-1)
    # err_pred = (err_pred - err_pred.mean())/err_pred.std()
    
    corr = torch.corrcoef(torch.stack([err_gt.view(-1).cuda(), err_pred.view(-1).cuda()]))
    # corr = torch.corrcoef(torch.stack([err_gt.sum(axis=(-1,-2)).cuda(), err_pred.sum(axis=(-1,-2)).cuda()]))
    print(type(pipeline.trainer.pipeline.model.renderer_entropy))
    print('corr:',corr[0,1].item())

    if return_images:
        return err_pred.detach().cpu(), err_gt.detach().cpu()
    else:
        return corr[0,1].item()
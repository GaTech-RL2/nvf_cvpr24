import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from nvf.env.utils import GIFSaver, get_images, pose_point_to
from nvf.visibility.visibility import *
# from nvf.uncertainty.entropy_renderers import VisibilityEntropyRenderer, WeightDistributionEntropyRenderer

def gen_data(cameras, model, num_samples):
    aabb = model.field.aabb
    samples = torch.rand(num_samples, 3, dtype=cameras.camera_to_worlds.dtype, device=cameras.camera_to_worlds.device)

    points = samples * (aabb[1] - aabb[0]) + aabb[0]
    visibility = get_visibility(points, cameras, model)

    # weight = get_balance_weight(visibility)
    return points, visibility#, weight

def get_balance_weight(visibility):
    mask = visibility>0.5

    weight = torch.ones_like(visibility)

    # return weight

    weight_pos = (mask*2.).mean()
    weight_neg = 2 - weight_pos
    

    if weight_pos<1e-7 or weight_neg<1e-7:
        return weight

    base = 0.3

    weight_pos = 1./weight_pos*(1-base)+base
    weight_neg = 1./weight_neg*(1-base)+base

    weight[mask] = weight_pos
    weight[~mask] = weight_neg

    # print(weight_pos, weight_neg)

    return weight

def train(cameras, pipeline):
    # num_train = 65536
    # pipeline.model.field.training = False
    batch_size = 65536 *4
    num_eval = batch_size
    epochs = 500
    train_batch_repeat = 5
    learning_rate = 1e-3

    # num_batch_train = num_train // batch_size
    num_batch_eval = num_eval // batch_size

    # train_points, train_visibility = gen_data(cameras, pipeline, num_train)
    eval_points, eval_visibility = gen_data(cameras, pipeline.model, num_eval)
    eval_weight = get_balance_weight(eval_visibility)
    model = pipeline.model.field.visibility_head
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    for i in range(epochs+1):
        torch.cuda.empty_cache()
        train_points, train_visibility = gen_data(cameras, pipeline.model, batch_size)
        train_weight = get_balance_weight(train_visibility)
        loss_fn.weight = train_weight

        train_points.requires_grad = False
        density_embedding = get_density_embedding(train_points, pipeline.model).detach()

        train_points.requires_grad = True
        density_embedding.requires_grad = True
        for _ in range(train_batch_repeat):
            optimizer.zero_grad()
            
            
            train_pred = pipeline.model.field.get_visibility(train_points, density_embedding, activation=False)
            
            train_loss = loss_fn(train_pred, train_visibility)
            train_loss.backward()
            optimizer.step()

        if i%100==0:
            eval_loss=0.
            for _ in range(num_batch_eval):
                density_embedding = get_density_embedding(eval_points, pipeline.model).detach()
                with torch.no_grad():
                    eval_pred = pipeline.model.field.get_visibility(eval_points, density_embedding, activation=False)
                loss_fn.weight = eval_weight
                eval_loss += loss_fn(eval_pred, eval_visibility)
            eval_loss /= num_batch_eval

            print(i, train_loss.item(), eval_loss.item())
             
    pass

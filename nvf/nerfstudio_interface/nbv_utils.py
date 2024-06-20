from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal, Optional, Tuple

import torch
import yaml

from nerfstudio.configs.method_configs import all_methods
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.engine.trainer import Trainer
from nerfstudio.engine.callbacks import TrainingCallbackAttributes


from nerfstudio.pipelines.base_pipeline import Pipeline

from nerfstudio.utils.rich_utils import CONSOLE

def load_checkpoint(load_path: os.PathLike, pipeline: Pipeline) -> None:
    """Helper function to load checkpointed pipeline
    Args:
        config (DictConfig): Configuration of pipeline to load
        pipeline (Pipeline): Pipeline instance of which to load weights
    Returns:
        A tuple of the path to the loaded checkpoint and the step at which it was saved.
    """
    # load_path = Path(load_path)
    CONSOLE.print("Loading checkpoint from path")
    loaded_state = torch.load(load_path, map_location="cpu")
    pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
    CONSOLE.print(f":white_check_mark: Done loading checkpoint from {load_path}")

def save_init_checkpoint(config_path: os.PathLike, trainer: Trainer) -> None:
    ckpt_path: Path = config_path / f"initial_weight.ckpt"
    torch.save(
        {
            "step": 1,
            # "pipeline": trainer.pipeline.state_dict(),
            "model": trainer.pipeline._model.state_dict(),
            "optimizers": {k: v.state_dict() for (k, v) in trainer.optimizers.optimizers.items()},
            "scalers": trainer.grad_scaler.state_dict(),
        },
        ckpt_path,
    )

def load_init_checkpoint(config_path: os.PathLike, trainer: Trainer) -> None:
    CONSOLE.print("Loading initial model weights")
    ckpt_path: Path = config_path / f"initial_weight.ckpt"
    loaded_state = torch.load(ckpt_path, map_location="cpu")

    trainer.pipeline._model.update_to_step(1)
    trainer._start_step = 1
    
    trainer.pipeline._model.load_state_dict(loaded_state["model"])
    trainer.optimizers.load_optimizers(loaded_state["optimizers"])
    # trainer.grad_scaler.load_state_dict(loaded_state["scalers"])

    trainer.callbacks = trainer.pipeline.get_training_callbacks(
        TrainingCallbackAttributes(
            optimizers=trainer.optimizers,
            grad_scaler=trainer.grad_scaler,
            pipeline=trainer.pipeline,
        )
    )

    del loaded_state

    CONSOLE.print(f":white_check_mark: Done loading initial model weights checkpoint from {ckpt_path}")

def reinitialize_weights(model):
    '''Reinitialize all weights of a PyTorch model using default initialization methods.'''
    for module in model.modules():
        if hasattr(module, 'reset_parameters'):
            print('reset', module)
            module.reset_parameters()

            
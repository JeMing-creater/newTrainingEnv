import os
import re
import random
import sys
import json
from collections import OrderedDict
from functools import wraps
import math
import numpy as np
import torch
from dataclasses import fields
from accelerate import Accelerator
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from torch import nn
from pathlib import Path
import numpy as np
import nibabel as nib
from transformers.trainer_utils import get_last_checkpoint, EvalPrediction

# compute loss and save log
def log_losses(logger, func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return_outputs = False
        if "return_outputs" in kwargs:
            return_outputs = kwargs["return_outputs"]
            kwargs.pop("return_outputs")
        # Call the original function
        loss, outputs = func(self, return_outputs=True, *args, **kwargs)
        # Log the loss fields in the outputs
        logger.info(
            {
                "best": self.state.best_metric,
                **{
                    field.name: f"{getattr(outputs, field.name).tolist()}"
                    for field in fields(outputs)
                    if field.name.startswith("loss")
                },
            }
        )

        return (loss, outputs) if return_outputs else loss

    return wrapper

# use for save checkpoint and load checkpoint
def set_checkpoint(logger, training_args):
    # checkpoint
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)  # set a checkpoint output dir in json
        and training_args.do_train    # check for training but not for testing or val
        and not training_args.overwrite_output_dir   # if set true, new training will overwrite old checkpoint and not loading checkpoint
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            logger.info(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
            # set for alread overwrite setting, if no checkpoint but overwrite is false will sent ValueError
            training_args.overwrite_output_dir = True
            training_args.resume_from_checkpoint = False
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
            training_args.overwrite_output_dir = True
            training_args.resume_from_checkpoint = False
    
    # use checkpoint for training resuming        
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    
    return checkpoint, training_args

# some notes will be writed in json, need to remove
def check_json(input_path: str) -> str:
    output_path = os.path.join(os.path.dirname(input_path), 'load.json')
    
    # load json
    with open(input_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # remove // 
    content_no_single_line_comments = re.sub(r'//.*', '', content, flags=re.MULTILINE)
    
    # remove /* and */ 
    content_cleaned = re.sub(r'/\*.*?\*/', '', content_no_single_line_comments, flags=re.DOTALL)

    # ensure json right
    try:
        json.loads(content_cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"{e}")
    
    # write new json
    output_dir = os.path.dirname(output_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(content_cleaned)
    
    return output_path

if __name__ == '__main__':
    json_path = '/workspace/Jeming/Project2/jsons/test.json'
    output_path = check_json(json_path)
    print(output_path)
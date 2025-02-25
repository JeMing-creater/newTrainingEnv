import os

# # debug in single GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import json
import monai
import torch
from accelerate.logging import get_logger
from accelerate.state import PartialState
from loguru import logger as loguru_logger


# HuggingFace tools
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.training_args import default_logdir
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl


# src functions
from src.loader import get_gcm_dataset
from src.utils import log_losses, set_checkpoint, check_json

# Arguments
from src.arg import DataArguments, ModelArguments

# models
from src.models.HWAUNETR.modeling import HWAUNETRForSegmentation, HWAUNETRConfig

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # For quickly link HuggingFace

# For this running time's log
logger = get_logger(__name__) 
Trainer.compute_loss = log_losses(logger, Trainer.compute_loss)

# For this running time's metrices
dice_metrics = monai.metrics.DiceMetric(include_background=True,reduction=monai.utils.MetricReduction.MEAN_BATCH, get_not_nans=True)
hd95_metrics = monai.metrics.HausdorffDistanceMetric(percentile=95, include_background=True,reduction=monai.utils.MetricReduction.MEAN_BATCH,get_not_nans=True)
dice_metrics.reset()
hd95_metrics.reset()

# Activation
post_trans = monai.transforms.Compose([
        monai.transforms.Activations(sigmoid=True), monai.transforms.AsDiscrete(threshold=0.5)
    ])

# Set a function for recall log info
class LogCallback(TrainerCallback):
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        logger.info(logs)


# Metrices computing for huggingface trainer
@torch.no_grad()
def compute_metrics(eval_pred: EvalPrediction, compute_result=False):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = post_trans(logits)
        dice_metrics(y_pred=logits_tensor, y=labels)
        hd95_metrics(y_pred=logits_tensor, y=labels)
        # pred_labels = logits_tensor.detach().cpu().numpy()
        
        # the last batch for an epoch, trainer will set compute_result to true and get scores
        if compute_result == True:
            dice_acc = dice_metrics.aggregate()[0]
            hd95_acc = hd95_metrics.aggregate()[0]
            scores = {
                f"Dice/Mean": float(dice_acc.mean()),
                f"Dice/ADC": float(dice_acc[0]),
                f"Dice/T2_FS": float(dice_acc[1]),
                f"Dice/V": float(dice_acc[2]),
                f"Hd95/Mean": float(hd95_acc.mean()),
                f"Hd95/ADC": float(hd95_acc[0]),
                f"Hd95/T2_FS": float(hd95_acc[1]),
                f"Hd95/V": float(hd95_acc[2]),
            }
            
    return scores if compute_result else None

# Training function
def train(
        training_args: TrainingArguments,
        model_args: ModelArguments,
        data_args: DataArguments,
    ):
    logger.info("initialize model")
    
    # model
    model = HWAUNETRForSegmentation(
        HWAUNETRConfig(
            in_chans = model_args.in_chans, 
            out_chans = model_args.out_chans, 
            fussion = model_args.fussion, 
            kernel_sizes = model_args.kernel_sizes, 
            depths = model_args.depths, 
            dims = model_args.dims, 
            heads = model_args.heads, 
            hidden_size = model_args.hidden_size, 
            num_slices_list = model_args.num_slices_list,
            out_indices = model_args.out_indices,
            d_state = model_args.d_state,
            d_conv = model_args.d_conv,
            expand = model_args.expand
        )
    )
    
    # dataset
    train_dataset, val_dataset, test_dataset = get_gcm_dataset(data_args=data_args)
    
    
    # checkpoint
    checkpoint = set_checkpoint(logger=logger, training_args=training_args)
    
    # training 
    logger.info("Starting training")
    trainer = Trainer(
        model,
        training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        callbacks=[LogCallback()],
    )
    
    trainer.train(resume_from_checkpoint=checkpoint)
    
    # Evaluate and save model 
    metirc = trainer.evaluate(test_dataset)
    logger.info(metirc)
    model.save_pretrained(f"{training_args.output_dir}/best")
    with open(f"{training_args.output_dir}/best/metric.json", "w") as f:
        json.dump(metirc, f, indent=4)
    
    

if __name__ == '__main__':
    
    # Environment setting
    torch.multiprocessing.set_sharing_strategy("file_system")
    parser = HfArgumentParser((TrainingArguments, ModelArguments, DataArguments))
    
    
    # choose json  
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        args = parser.parse_json_file(check_json(os.path.abspath(sys.argv[1]))) # set for os cmd input
    else:
        args = parser.parse_json_file(check_json("jsons/training_config_gcm.json"))  # base setting
        
        
    # Note: Parameter The following arg parameters need to be unique and correspond to the key values in json.
    training_args: TrainingArguments = args[0]
    model_args: ModelArguments = args[1]
    data_args: DataArguments = args[2]
    
    
    # # Identify FOL from ENV, if experiment need a k-fold test, please set a key value in data arg.
    # kfold = os.getenv("KFOLD")
    # if kfold:
    #     data_args.kfold = int(kfold)
    # training_args.output_dir = f"{training_args.output_dir}_fold{data_args.kfold}"
    
    
    # set logging dir
    training_args.logging_dir = os.path.join(training_args.output_dir, default_logdir())
    if PartialState().is_main_process:
        loguru_logger.add(f"{training_args.logging_dir}/log.txt")
    
    # training
    train(training_args, model_args, data_args)
    
    
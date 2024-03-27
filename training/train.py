#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 SPRIGHT authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse

# added for gaudi
import json
import logging
import math
import os
import random
import shutil
import time
from pathlib import Path

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator


try:
    from optimum.habana import GaudiConfig
    from optimum.habana.accelerate import GaudiAccelerator
except:
    GaudiConfig = None
    GaudiAccelerator = None

from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration


try:
    from optimum.habana.utils import set_seed
except:
    from accelerate.utils import set_seed
import datetime

from datasets import DownloadMode, load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import diffusers
from diffusers import AutoencoderKL, UNet2DConditionModel


try:
    from optimum.habana.diffusers import GaudiDDIMScheduler, GaudiStableDiffusionPipeline
except:
    from diffusers import DDPMScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import deprecate, is_wandb_available, make_image_grid


try:
    # memory stats
    import habana_frameworks.torch as htorch
    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.hpu as hthpu
except:
    from diffusers.utils.import_utils import is_xformers_available
    htorch = None
    hthpu = None
    htcore = None
import sys


sys.path.append(os.path.dirname(os.getcwd()))
import itertools
import warnings

import webdataset as wds
from transformers import PretrainedConfig

from diffusers.utils.torch_utils import is_compiled_module


if is_wandb_available():
    import wandb

debug = False

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
#check_min_version("0.23.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def save_model_card(
    args,
    repo_id=None,
    images=None,
    train_text_encoder=False,
    repo_folder=None,
):
    img_str = ""
    if len(images) > 0:
        image_grid = make_image_grid(images, 1, len(args.validation_prompts))
        image_grid.save(os.path.join(repo_folder, "val_imgs_grid.png"))
        img_str += "![val_imgs_grid](./val_imgs_grid.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {args.pretrained_model_name_or_path}
datasets:
- {args.dataset_name}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
inference: true
---
    """
    model_card = f"""
# Text-to-image finetuning - {repo_id}
Fine-tuning for the text encoder was enabled: {train_text_encoder}.

This pipeline was finetuned from **{args.pretrained_model_name_or_path}** on the **{args.dataset_name}** dataset. Below are some example images generated with the finetuned pipeline using the following prompts: {args.validation_prompts}: \n
{img_str}

## Pipeline usage

You can use the pipeline like so:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("{repo_id}", torch_dtype=torch.float16)
prompt = "{args.validation_prompts[0]}"
image = pipeline(prompt).images[0]
image.save("my_image.png")
```

## Training info

These are the key hyperparameters used during training:

* Epochs: {args.num_train_epochs}
* Learning rate: {args.learning_rate}
* Batch size: {args.train_batch_size}
* Gradient accumulation steps: {args.gradient_accumulation_steps}
* Image resolution: {args.resolution}
* Mixed-precision: {args.mixed_precision}

"""
    wandb_info = ""
    if is_wandb_available():
        wandb_run_url = None
        if wandb.run is not None:
            wandb_run_url = wandb.run.url

    if wandb_run_url is not None:
        wandb_info = f"""
More information on all the CLI arguments and the environment are available on your [`wandb` run page]({wandb_run_url}).
"""

    model_card += wandb_info

    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)

def compute_validation_loss(val_dataloader, vae, text_encoder, noise_scheduler, unet, args, weight_dtype):
    val_loss = 0
    num_steps= math.ceil(len(val_dataloader))
    progress_bar = tqdm(
        range(0, num_steps),
        initial=0,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=True,
    )
    for step, batch in enumerate(val_dataloader):
        progress_bar.update(1)
        # Convert images to latent space
        latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        if args.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += args.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
            )
        if args.input_perturbation:
            new_noise = noise + args.input_perturbation * torch.randn_like(noise)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        if args.input_perturbation:
            noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
        else:
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = text_encoder(batch["input_ids"])[0]

        # Get the target for loss depending on the prediction type
        if args.prediction_type is not None:
            # set prediction_type of scheduler if defined
            noise_scheduler.register_to_config(prediction_type=args.prediction_type)

        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        # Predict the noise residual and compute loss
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        if args.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(noise_scheduler, timesteps)
            if noise_scheduler.config.prediction_type == "v_prediction":
                # Velocity objective requires that we add one to SNR values before we divide by them.
                snr = snr + 1
            mse_loss_weights = (
                torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        logs = {"step_val_loss": loss.detach().item()}
        progress_bar.set_postfix(**logs)
        val_loss += loss.item()
    val_loss /= (step+1)
    return val_loss

def log_validation(vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, epoch, val_dataloader, noise_scheduler):
    logger.info("Running validation... ")


    if args.validation_prompts is not None:

        if args.device == "hpu":
            pipeline = GaudiStableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=accelerator.unwrap_model(text_encoder),
                tokenizer=tokenizer,
                vae=accelerator.unwrap_model(vae),
                unet=accelerator.unwrap_model(unet),
                safety_checker=None,
                revision=args.revision,
                use_habana=True,
                use_hpu_graphs=True,
                gaudi_config=args.gaudi_config_name,
            )
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                vae=accelerator.unwrap_model(vae),
                text_encoder=accelerator.unwrap_model(text_encoder),
                tokenizer=tokenizer,
                unet=accelerator.unwrap_model(unet),
                safety_checker=None,
                revision=args.revision,
                torch_dtype=weight_dtype,
            )
            pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        if args.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()

        if args.seed is None:
            generator = None
        else:
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

        images = []
        for i in range(len(args.validation_prompts)):
            if args.device == "hpu":
                image = pipeline(args.validation_prompts[i], num_inference_steps=50, generator=generator).images[0]
            else:
                with torch.autocast("cuda"):
                    image = pipeline(args.validation_prompts[i], num_inference_steps=50, generator=generator).images[0]

            images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            if args.validation_prompts is not None:
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images("validation/images", np_images, epoch, dataformats="NHWC")
        elif tracker.name == "wandb":
            if args.validation_prompts is not None:
                tracker.log(
                    {
                        "validation/images": [
                            wandb.Image(image, caption=f"{i}: {args.validation_prompts[i]}")
                            for i, image in enumerate(images)
                        ]
                    }
                )

        else:
            if args.device == "hpu":
                logger.warning(f"image logging not implemented for {tracker.name}")
            else:
                logger.warn(f"image logging not implemented for {tracker.name}")

    del pipeline
    if args.device != "hpu":
        torch.cuda.empty_cache()

    return images

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--spright_splits",
        type=str,
        default="split.json",
        help=(
            "A url containing the json file that defines the splits (https://huggingface.co/datasets/ASU-HF/spright/blob/main/split.json). The webdataset should contain the metadata as tar files."
        ),
    )
    parser.add_argument(
        "--spright_train_costum",
        type=str,
        default=None,
        help=(
            "A url containing the webdataset train split. The webdataset should contain the metadata as tar files."
        ),
    )
    parser.add_argument(
        "--spright_val_costum",
        type=str,
        default=None,
        help=(
            "A url containing the webdataset validation split. The webdataset should contain the metadata as tar files."
        ),
    )
    parser.add_argument(
        "--webdataset_buffer_size",
        type=int,
        default=1000,
        help=(
            "buffer size of webdataset."
        ),
    )
    parser.add_argument(
        "--dataset_size",
        type=float,
        default=None,
        help="dataset size to use. If set, the dataset will be truncated to this size.",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="ratio of validation size out of the entire dataset "
    )
    parser.add_argument(
        "--train_metadata_dir",
        type=str,
        default=None,
        help=(
            "A folder containing subfolders: train, val, test with the metadata as jsonl files."
            " jsonl files provide the general and spatial captions for the images."
        ),
    )
    parser.add_argument(
        "--dataloader",
        type=str,
        default=None,
        help=(
            "A python script with custom dataloader."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=["The city is located behind the water, and the pier is relatively small in comparison to the expanse of the water and the city", "The bed is positioned in the center of the frame, with two red pillows on the left side", "The houses are located on the left side of the street, while the park is on the right side", "The spoon is located on the left side of the shelf, while the bowl is positioned in the center", "The room has a red carpet, and there is a chandelier hanging from the ceiling above the bed"],
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--pre_crop_resolution",
        type=int,
       default=768,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution before being randomly cropped to the final `resolution`."
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=None,
        help="Initial learning rate for the text encoder - should usually be samller than unet_lr(after the potential warmup period) to use. When set to None, it will be set to the same value as learning_rate.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--gaudi_config_name",
        type=str,
        default=None,
        help="Local path to the Gaudi configuration file or its name on the Hugging Face Hub.",
    )
    parser.add_argument(
        "--throughput_warmup_steps",
        type=int,
        default=0,
        help=(
            "Number of steps to ignore for throughput calculation. For example, with throughput_warmup_steps=N, the"
            " first N steps will not be considered in the calculation of the throughput. This is especially useful in"
            " lazy mode."
        ),
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=False,
        help=("Whether to use bf16 mixed precision."),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=("hpu, cuda or cpu."),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--freeze_text_encoder_steps",
        type=int,
        default=0,
        help="Start text_encoder training after freeze_text_encoder_steps steps.",
    )
    parser.add_argument("--comment", type=str, default="used long sentences generated by llava - spatial and general.", help="Comment that should appear in the run config")
    parser.add_argument("--git_token", type=str, default=None, help="If provided will enable to save the git sha to replicate")
    parser.add_argument("--general_caption", type=str, default="original_caption", choices = ["coca_caption", "original_caption"],
                        help="Original are the oned from the original dataset, coca_caption is the one generated by COCA" \
                            "in case original is chosen, the original caption will be preffered as general_caption, if it does not exist than the general caption will be the coca caption")
    parser.add_argument("--spatial_caption_type", type=str, required=True, choices = ["short", "long", "short_negative"], help="Wheter to use long or short spatial captions")
    parser.add_argument(
        "--spatial_percent",
        type=float,
        default=50.0,
        help="approximately precentage of the time that spatial captions is chosen.",
    )


    args = parser.parse_args()
    if args.resume_from_checkpoint is None:
        args.output_dir = os.path.join(args.output_dir , f"run_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():
    args = parse_args()

    url_train = None

    if args.spright_splits is not None and args.spright_train_costum is not None:
        warnings.warn("You can not specify the splits by both spright_splits and spright_train_costum." \
                      "The costum split will be used. If you want to use the SPRIGHT splits, remove the spright_train_costum argument.")


    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    # set device
    if hthpu and hthpu.is_available():
        args.device = "hpu"
        logger.info("Using HPU")
    elif torch.cuda.is_available():
        logger.info.device = "cuda"
        print("Using GPU")
    else:
        args.device = "cpu"
        logger.info("Using CPU")

    # set precision:
    if args.device == "hpu":
        if args.mixed_precision == "bf16":
            args.bf16 = True
        else:
            args.bf16 = False

        # set args for gaudi:
        assert not args.enable_xformers_memory_efficient_attention, "xformers is not supported on gaudi"
        assert not args.allow_tf32, "tf32 is not supported on gaudi"
        assert not args.gradient_checkpointing, "gradient_checkpointing is not supported on gaudi locally"
        assert not args.push_to_hub, "push_to_hub is not supported on gaudi locally"

    else:
        assert args.gaudi_config_name is None, "gaudi_config_name is only supported on gaudi"
        assert args.throughput_warmup_steps == 0, "throughput_warmup_steps is only supported on gaudi"


    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    if args.device == "hpu":
        gaudi_config = GaudiConfig.from_pretrained(args.gaudi_config_name)
        if args.use_8bit_adam:
            gaudi_config.use_fused_adam = True
            args.use_8bit_adam = False

        accelerator = GaudiAccelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision="bf16" if gaudi_config.use_torch_autocast or args.bf16 else "no",
            log_with=args.report_to,
            project_config=accelerator_project_config,
            force_autocast=gaudi_config.use_torch_autocast or args.bf16,
        )
    else:
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_config=accelerator_project_config,
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
    if args.device == "hpu":
        noise_scheduler = GaudiDDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    else:
        noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    if not args.train_text_encoder:
        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            text_encoder = CLIPTextModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
            ).to(accelerator.device)
            vae = AutoencoderKL.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
            )
    else:
        # import correct text encoder class
        text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
        text_encoder = text_encoder_cls.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
        )


    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )

    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)
        unet.train()

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
        )
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for model in models:
                    sub_dir = "unet" if isinstance(model, type(unwrap_model(unet))) else "text_encoder"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                if isinstance(model, type(unwrap_model(text_encoder))):
                    # load transformers style into model
                    load_model = text_encoder_cls.from_pretrained(input_dir, subfolder="text_encoder")
                    model.config = load_model.config
                else:
                    # load diffusers style into model
                    load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                    model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    if unwrap_model(unet).dtype != torch.float32:
        raise ValueError(f"Unet loaded as datatype {unwrap_model(unet).dtype}. {low_precision_error_string}")

    if args.train_text_encoder and unwrap_model(text_encoder).dtype != torch.float32:
        raise ValueError(
            f"Text encoder loaded as datatype {unwrap_model(text_encoder).dtype}." f" {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    elif args.device == "hpu" and gaudi_config.use_fused_adam:
        from habana_frameworks.torch.hpex.optimizers import FusedAdamW

        optimizer_cls = FusedAdamW
    else:
        optimizer_cls = torch.optim.AdamW
    # setting diffetent lr for text ancoder and unet:
    if args.train_text_encoder:
        unet_parameters_with_lr = {"params": unet.parameters(), "lr": args.learning_rate}
        text_encoder_params_with_lr = {
            "params": text_encoder.parameters(),
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate
        }
        params_to_optimize = [
            unet_parameters_with_lr,
            text_encoder_params_with_lr
        ]
        print(f"Learning rates were provided both for the unet and the text encoder- e.g. text_encoder_lr:"
              f" {args.text_encoder_lr} and learning_rate: {args.learning_rate}. ")


    else:
        params_to_optimize = unet_parameters_with_lr
    optimizer = optimizer_cls(
        params_to_optimize,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )



    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            data_dir=args.train_data_dir,
            download_mode=DownloadMode.FORCE_REDOWNLOAD
        )
    elif args.train_data_dir is not None:
        data_files = {}
        data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder
    elif args.dataloader is not None:
        dataset = load_dataset(
            args.dataloader,
            data_dir=args.train_metadata_dir,
            cache_dir=args.cache_dir
        )
    elif args.spright_splits is not None:
        # load the jaon files and read the train and val splits
        if args.spright_splits.endswith(".json"):
            with open(args.spright_splits, 'r') as f:
                data = json.load(f)
            # Filter the entries where split is 'train'
            train_files = []
            train_len = 0
            val_files = []
            val_len = 0
            test_files = []
            test_len = 0
            if args.spright_train_costum is not None:
                url_train = args.spright_train_costum
                start, end = map(int, url_train.split('/')[-1].strip('.tar{}').split('..'))
                # Filter the data for the files in the range and sum the sizes
                train_len = sum(item['size'] for item in data if start <= int(item['file'].strip('.tar')) <= end)
                if args.spright_val_costum is not None:
                    url_val = args.spright_val_costum
                    start_val, end_val = map(int, url_train.split('/')[-1].strip('.tar{}').split('..'))
                    val_len = sum(item['size'] for item in data if start_val <= int(item['file'].strip('.tar')) <= end_val)
                else:
                    url_val = None
            else:
                for item in data:
                    if item['split'] == 'train':
                        train_files.append(item['file'].split(".")[0])
                        train_len += item['size']
                    elif item['split'] == 'val':
                        val_files.append(item['file'].split(".")[0])
                        val_len += item['size']
                    elif item['split'] == 'test':
                        test_files.append(item['file'].split(".")[0])
                        test_len += item['size']
                ext = data[0]['file'].split(".")[-1]
                # Construct the url_train string
                if len(train_files) != 0:
                    url_train = "/export/share/projects/mcai/spatial_data/spright/data/{" + ','.join(train_files) + "}" + f".{ext}"
                if len(val_files) != 0:
                    url_val = "/export/share/projects/mcai/spatial_data/spright/data/{" + ','.join(val_files) + "}" + f".{ext}"
                else:
                    url_val = None
        else:
            raise ValueError("'webdataset' should be a json file containing the train and val splits and there sizes")

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    # column_names = dataset["train"].column_names
    # 6. Get the column names for input/target.
    if args.spatial_caption_type == "short":
        spatial_caption = 'short_spatial_caption'
    elif args.spatial_caption_type == "long":
        spatial_caption = 'spatial_caption'
    elif args.spatial_caption_type == "short_negative":
        spatial_caption = 'short_spatial_caption_negation'

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.

    def get_random_sentence(caption):
        if isinstance(caption, list):
            caption = caption[0]
        sentences_split = caption.split(".")
        first_sentence = sentences_split[0]
        sentences_split = sentences_split[1:]
        if len(sentences_split)>1:
            random_sentence = random.choice(sentences_split)
            if len(random_sentence) == 0:
                random_sentence = sentences_split[0]
        else:
            random_sentence = first_sentence
        return random_sentence


    def tokenize_captions(examples, is_train=True):
        # check if its a webdataset
        if url_train is not None:
            examples = examples["captions"]

        column_lists = [args.general_caption, 'spatial_caption']

        caption_column = random.choices(
            column_lists,
            weights=[100-args.spatial_percent, args.spatial_percent],
            k=1
        )[0]

        # check if the caption exists
        if args.general_caption=="original_caption" and caption_column==args.general_caption:
            if caption_column not in examples.keys():
                caption_column = "coca_caption"
        elif args.spatial_caption_type == "short" and caption_column==spatial_caption:
            examples['short_spatial_caption'] = get_random_sentence(examples[caption_column])
            caption_column = 'short_spatial_caption'

        captions = []
        # check if the caption is a list
        if not isinstance(examples[caption_column], list):
            examples[caption_column] = [examples[caption_column]]

        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
                if debug:
                    with open(os.path.join(args.output_dir, f"selected_captions{args.spatial_percent}.txt"), "a") as f:
                        f.write(f"{caption}\n")
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    if args.pre_crop_resolution and args.pre_crop_resolution > args.resolution:
        train_transforms = transforms.Compose(
            [
                transforms.Resize(args.pre_crop_resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
                transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    else:
        train_transforms = transforms.Compose(
            [
                transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
                transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def preprocess_train(examples):
        if url_train is not None:
            examples["image"] = [examples["image"]]
        images = [image.convert("RGB") for image in examples["image"]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    def preprocess_train_back(examples):
        images = [image.convert("RGB") for image in examples["image"]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    def collate_fn_webdataset(examples):
        examples = [preprocess_train(example) for example in examples]
        pixel_values = torch.stack([example["pixel_values"][0] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    with accelerator.main_process_first():
        if url_train is None:
            if args.max_train_samples is not None:
                dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
            train_dataset = dataset["train"].with_transform(preprocess_train)
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                shuffle=True,
                collate_fn=collate_fn,
                batch_size=args.train_batch_size,
                num_workers=args.dataloader_num_workers,
            )
            if "validation" in dataset:
                val_dataloader = torch.utils.data.DataLoader(
                    dataset["validation"].with_transform(preprocess_train),
                    shuffle=True,
                    collate_fn=collate_fn,
                    batch_size=args.train_batch_size,
                    num_workers=0,
                )
            else:
                val_dataloader = None
        else:
            dataset = {"train": wds.WebDataset(url_train).shuffle(args.webdataset_buffer_size).decode("pil", handler=wds.warn_and_continue).rename(captions="json",image="jpg",metadata="metadata.json",handler=wds.warn_and_continue,)}
            train_dataloader = torch.utils.data.DataLoader(dataset["train"], collate_fn=collate_fn_webdataset, batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers)
            if url_val:
                dataset["val"] = wds.WebDataset(url_val).shuffle(args.webdataset_buffer_size).decode("pil", handler=wds.warn_and_continue). rename(captions="json",image="jpg",metadata="metadata.json",handler=wds.warn_and_continue,)
                val_dataloader = torch.utils.data.DataLoader(dataset["val"], collate_fn=collate_fn_webdataset, batch_size=args.train_batch_size, num_workers=0)
            else:
                val_dataloader = None
            if args.max_train_samples is not None:
                dataset = (
                    wds.WebDataset(url_train, shardshuffle=True)
                    .shuffle(args.webdataset_buffer_size)
                    .decode()
                )
            # Set the training transforms
            train_dataset = dataset["train"]

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    if url_train is None:
        train_len = len(train_dataset)
        train_dataloader_len = len(train_dataloader)
        assert len(train_dataloader) == len(train_dataset) / args.train_batch_size, (len(train_dataloader), len(train_dataset))
    else:
        train_dataloader_len = train_len / args.train_batch_size

    num_update_steps_per_epoch = math.ceil(train_dataloader_len / args.gradient_accumulation_steps)

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    if not args.train_text_encoder:
        unet.to(accelerator.device)

    # Prepare everything with our `accelerator`.
    if val_dataloader is not None:
        if args.train_text_encoder:
            unet, text_encoder, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
                unet, text_encoder, optimizer, train_dataloader,val_dataloader, lr_scheduler
            )
        else:
            unet, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
                unet, optimizer, train_dataloader, val_dataloader, lr_scheduler
            )
    else:
        if args.train_text_encoder:
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, text_encoder, optimizer, train_dataloader, lr_scheduler
            )
        else:
            unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, optimizer, train_dataloader, lr_scheduler
            )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if args.device != "hpu" and accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision
    elif args.device == "hpu" and gaudi_config.use_torch_autocast or args.bf16:
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    if not args.train_text_encoder and text_encoder is not None:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(train_dataloader_len / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {train_len}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if accelerator.is_main_process:
        run_config = vars(args)
        with open(os.path.join(args.output_dir, "run_config.jsonl"), 'a') as f:
            for key, value in run_config.items():
                json.dump({key: value}, f)
                f.write('\n')

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        print(f"output_dir used for resuming is: {args.output_dir}")
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    if args.device == "hpu":
        t0 = None

    # saving the model before training
    if args.device == "hpu":
        pipeline = GaudiStableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=unwrap_model(text_encoder) if args.train_text_encoder else text_encoder,
            vae=vae,
            unet=unwrap_model(unet),
            revision=args.revision,
            scheduler=noise_scheduler,
        )
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=unwrap_model(text_encoder) if args.train_text_encoder else text_encoder,
            vae=vae,
            unet= unwrap_model(unet),
            revision=args.revision,
        )
    pipeline.save_pretrained(args.output_dir)
    if accelerator.is_main_process:
        log_validation(
            vae,
            text_encoder,
            tokenizer,
            unet,
            args,
            accelerator,
            weight_dtype,
            global_step,
            val_dataloader,
            noise_scheduler,
        )
    text_train_active = False
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.train_text_encoder and global_step > args.freeze_text_encoder_steps:
            text_encoder.train()
        train_loss = 0.0
        print("epoch: ", epoch)
        for step, batch in enumerate(train_dataloader):
            if args.train_text_encoder and global_step > args.freeze_text_encoder_steps and not text_train_active:
                text_encoder.train()
                text_train_active = True
                print("Text encoder training started at {} steps".format(global_step))

            if args.device == "hpu":
                if t0 is None and global_step == args.throughput_warmup_steps:
                    t0 = time.perf_counter()

            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                if args.input_perturbation:
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # write prediction_type to run_config.json
                if accelerator.is_main_process and epoch == 0 and step == 0:
                    with open(os.path.join(args.output_dir, "run_config.jsonl"), 'a') as f:
                        f.write(json.dumps({"prediction_type": noise_scheduler.config.prediction_type}) + '\n')

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if args.train_text_encoder and global_step > args.freeze_text_encoder_steps
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()

                if args.device == "hpu":
                    optimizer.zero_grad(set_to_none=True)
                    htcore.mark_step()
                else:
                    optimizer.zero_grad()
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"training/train_loss": train_loss}, step=global_step)
                accelerator.log({"hyperparameters/batch_size": args.train_batch_size}, step=global_step)
                accelerator.log({"hyperparameters/effective_batch_size": total_batch_size}, step=global_step)
                accelerator.log({"hyperparameters/learning_rate": lr_scheduler.get_last_lr()[0]}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if args.device == "hpu":
            #gg gaudi addition
            duration = time.perf_counter() - t0
            throughput = args.max_train_steps * total_batch_size / duration

        if accelerator.is_main_process:
            if args.device == "hpu":
                logger.info(f"Throughput = {throughput} samples/s")
                logger.info(f"Train runtime = {duration} seconds")
                metrics = {
                    "train_samples_per_second": throughput,
                    "train_runtime": duration,
                }
                with open(f"{args.output_dir}/speed_metrics.json", mode="w") as file:
                    json.dump(metrics, file)

            if epoch % args.validation_epochs == 0:
                if args.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())
                log_validation(
                    vae,
                    text_encoder,
                    tokenizer,
                    unet,
                    args,
                    accelerator,
                    weight_dtype,
                    global_step,
                    val_dataloader,
                    noise_scheduler,
                )
                if args.use_ema:
                    # Switch back to the original UNet parameters.
                    ema_unet.restore(unet.parameters())

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.device == "hpu":
            logger.info(f"Throughput = {throughput} samples/s")
            logger.info(f"Train runtime = {duration} seconds")
            metrics = {
                "train_samples_per_second": throughput,
                "train_runtime": duration,
            }
            with open(f"{args.output_dir}/speed_metrics.json", mode="w") as file:
                json.dump(metrics, file)

        unet = accelerator.unwrap_model(unet)
        if args.train_text_encoder:
            text_encoder = unwrap_model(text_encoder)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        if args.device == "hpu":
            pipeline = GaudiStableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
                revision=args.revision,
                scheduler=noise_scheduler,
            )
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
                revision=args.revision,
            )
        pipeline.save_pretrained(args.output_dir)
        if args.use_ema:
            ema_unet.save_pretrained(os.path.join(args.output_dir, "unet_ema"))

        # Run a final round of inference.
        images = []
        if args.validation_prompts is not None:
            logger.info("Running inference for collecting generated images...")
            pipeline = pipeline.to(accelerator.device)
            pipeline.torch_dtype = weight_dtype
            pipeline.set_progress_bar_config(disable=True)

            if args.enable_xformers_memory_efficient_attention:
                pipeline.enable_xformers_memory_efficient_attention()

            if args.seed is None:
                generator = None
            else:
                generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

            for i in range(len(args.validation_prompts)):
                if args.device == "hpu":
                    image = pipeline(args.validation_prompts[i], num_inference_steps=20, generator=generator).images[0]
                else:
                    with torch.autocast("cuda"):
                        image = pipeline(args.validation_prompts[i], num_inference_steps=20, generator=generator).images[0]
                images.append(image)

        # name of last directory in output path
        run_name = os.path.basename(os.path.normpath(args.output_dir))
        #save_model_card(args, None, images, repo_folder=args.output_dir)
        if args.push_to_hub:
            #save_model_card(args, repo_id, images, repo_folder=args.output_dir)
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                path_in_repo=f"gpu_runs/{run_name}",
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*, checkpoint-*"],
                allow_patterns=["checkpoint-15000"]
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
# limitations under the License.

"""Script to fine-tune PixArt-Alpha (DiT) for RSEdit (Token Concatenation)."""

import argparse
import logging
import math
import os
import shutil
import json
from contextlib import nullcontext
from pathlib import Path

import accelerate
import datasets
import numpy as np
import PIL
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import T5EncoderModel, T5Tokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DPMSolverMultistepScheduler, PixArtTransformer2DModel, PixArtAlphaPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.constants import DIFFUSERS_REQUEST_TIMEOUT
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from dataset import RSCCDataset
from pipeline_rsedit_dit import RSEditDiTPipeline


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.36.0.dev0")

logger = get_logger(__name__, log_level="INFO")

# DATASET_NAME_MAPPING = {
#     "fusing/instructpix2pix-1000-samples": ("input_image", "edit_prompt", "edited_image"),
# }
WANDB_TABLE_COL_NAMES = ["original_image", "edited_image", "edit_prompt"]


def log_validation(
    vae,
    text_encoder,
    tokenizer,
    transformer,
    scheduler,
    args,
    accelerator,
    generator,
    validation_items=None,
    guidance_scale=4.5,
):
    """Run validation using RSEditDiTPipeline."""
    if validation_items is None:
        if args.val_image_url and args.validation_prompt:
            validation_items = [(args.val_image_url, args.validation_prompt)]
        else:
            return

    logger.info(f"Running validation on {len(validation_items)} images... (CFG scale: {guidance_scale})")
    logger.info(f"Using seed: {args.seed} for reproducible validation")

    # Unwrap models if they're wrapped by Accelerator (e.g., DDP, compiled)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    
    # Unwrap transformer (it's prepared with accelerator.prepare())
    unwrapped_transformer = unwrap_model(transformer)
    
    # CRITICAL: Switch to AttnProcessor (non-SDPA) to avoid cuDNN errors
    # The default AttnProcessor2_0 uses F.scaled_dot_product_attention which
    # fails with cuDNN on non-standard sequence lengths from token concatenation
    from diffusers.models.attention_processor import AttnProcessor
    original_attn_processors = unwrapped_transformer.attn_processors
    unwrapped_transformer.set_attn_processor(AttnProcessor())
    
    # Create RSEdit pipeline with DPMSolverMultistepScheduler for inference
    pipeline_scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config)
    
    pipeline = RSEditDiTPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=unwrapped_transformer,
        scheduler=pipeline_scheduler,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # Setup wandb logging
    wandb_table = None
    if is_wandb_available():
        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                wandb_table = wandb.Table(columns=WANDB_TABLE_COL_NAMES)
                break

    # Create validation output directory
    val_save_dir = os.path.join(args.output_dir, "validation_images")
    if accelerator.is_main_process:
        os.makedirs(val_save_dir, exist_ok=True)

    # Run validation on each image/prompt pair
    image_idx = 0
    for image_path, prompt in validation_items:
        logger.info(f"Validating: {image_path} | Prompt: {prompt}")
        original_image = download_image(image_path)
        
        # Extract image name from path for saving
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Generate edited images using the RSEdit pipeline
        # Create a new generator with the same seed for reproducibility
        # Use CPU generator for better reproducibility across platforms
        val_generator = torch.Generator(device="cpu").manual_seed(args.seed)
        edited_images = pipeline(
            prompt=prompt,
            source_image=original_image,
            negative_prompt="",
            num_inference_steps=50,
            guidance_scale=guidance_scale,
            num_images_per_prompt=args.num_validation_images,
            height=args.resolution,
            width=args.resolution,
            generator=val_generator,
            clean_caption=False,
            max_sequence_length=120,
        ).images

        # Save original image (only once per image)
        if accelerator.is_main_process:
            original_save_path = os.path.join(val_save_dir, f"{image_idx:04d}_{image_name}_original.png")
            original_image.save(original_save_path)
            
            # Log original to TensorBoard (only once per image)
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    import numpy as np
                    tracker.writer.add_image(
                        f"validation/{image_idx:04d}_{image_name}/original",
                        np.array(original_image),
                        global_step=accelerator.step,
                        dataformats='HWC'
                    )
                    tracker.writer.add_text(
                        f"validation/{image_idx:04d}_{image_name}/prompt",
                        prompt,
                        global_step=accelerator.step
                    )

        # Log to wandb and save/log edited images
        for sample_idx, edited_image in enumerate(edited_images):
            if wandb_table is not None:
                wandb_table.add_data(
                    wandb.Image(original_image), wandb.Image(edited_image), prompt
                )
            
            # Save edited images locally
            if accelerator.is_main_process:
                edited_save_path = os.path.join(val_save_dir, f"{image_idx:04d}_{image_name}_edited_{sample_idx}.png")
                edited_image.save(edited_save_path)
                
                # Log to TensorBoard
                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        import numpy as np
                        tracker.writer.add_image(
                            f"validation/{image_idx:04d}_{image_name}/edited_{sample_idx}",
                            np.array(edited_image),
                            global_step=accelerator.step,
                            dataformats='HWC'
                        )
        
        image_idx += 1

    # Upload wandb table
    for tracker in accelerator.trackers:
        if tracker.name == "wandb" and wandb_table is not None:
            tracker.log({"validation": wandb_table})
    
    # Restore original attention processors for training
    unwrapped_transformer.set_attn_processor(original_attn_processors)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for RSEdit DiT.")
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
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
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
        "--original_image_column",
        type=str,
        default="input_image",
        help="The column of the dataset containing the original image on which edits where made.",
    )
    parser.add_argument(
        "--edited_image_column",
        type=str,
        default="edited_image",
        help="The column of the dataset containing the edited image.",
    )
    parser.add_argument(
        "--edit_prompt_column",
        type=str,
        default="edit_prompt",
        help="The column of the dataset containing the edit instruction.",
    )
    parser.add_argument(
        "--val_image_url",
        type=str,
        default=None,
        help="URL to the original image that you would like to edit (used during inference for debugging purposes).",
    )
    parser.add_argument(
        "--validation_prompt", type=str, default=None, help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=None,
        help=(
            "Run fine-tuning validation every X steps. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--val_set_path",
        type=str,
        default=None,
        help="Path to a text file containing paths of validation images.",
    )
    parser.add_argument(
        "--val_annotation_path",
        type=str,
        default=None,
        help="Path to the JSONL file containing captions for validation images.",
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
        "--output_dir",
        type=str,
        default="rsedit-dit-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
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
        "--conditioning_dropout_prob",
        type=float,
        default=None,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--use_prodigy", action="store_true", help="Whether to use Prodigy optimizer.")
    parser.add_argument("--prodigy_d_coef", type=float, default=1.0, help="Prodigy D coefficient (default: 1.0).")
    parser.add_argument("--prodigy_d0", type=float, default=1e-5, help="Prodigy initial D estimate (default: 1e-5).")
    parser.add_argument("--prodigy_safeguard_warmup", action="store_true", help="Use safeguard warmup for Prodigy.")
    parser.add_argument("--prodigy_use_bias_correction", action="store_true", help="Use bias correction for Prodigy.")
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

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def convert_to_np(image, resolution):
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)


def download_image(url):
    if os.path.exists(url):
        return PIL.Image.open(url).convert("RGB")
    image = PIL.Image.open(requests.get(url, stream=True, timeout=DIFFUSERS_REQUEST_TIMEOUT).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def load_validation_data(val_set_path, annotation_path):
    path_to_caption = {}
    with open(annotation_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    path_to_caption[item["pre_image"]] = item["change_caption"]
                except:
                    continue

    validation_items = []
    with open(val_set_path, "r") as f:
        for line in f:
            path = line.strip()
            if path and path in path_to_caption:
                validation_items.append((path, path_to_caption[path]))
            elif path:
                logger.warning(f"Validation image {path} not found in annotations file.")

    return validation_items


def main():
    args = parse_args()
    
    validation_items = None
    if args.val_set_path and args.val_annotation_path:
        validation_items = load_validation_data(args.val_set_path, args.val_annotation_path)

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `hf auth login` to authenticate with the Hub."
        )

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
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Create generator for training (conditioning dropout)
    # Use device-specific generator for training performance
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

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
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = T5Tokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    transformer = PixArtTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.non_ema_revision
    )

    logger.info("DiT Transformer loaded. Using Token Concatenation strategy.")
    # No architectural changes needed for Token Concatenation in PixArt/DiT 
    # as they handle sequence length flexibly via patchification.

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Create EMA for the transformer.
    if args.use_ema:
        ema_transformer = EMAModel(transformer.parameters(), model_cls=PixArtTransformer2DModel, model_config=transformer.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            transformer.enable_xformers_memory_efficient_attention()
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
                    ema_transformer.save_pretrained(os.path.join(output_dir, "transformer_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "transformer"))

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "transformer_ema"), PixArtTransformer2DModel)
                ema_transformer.load_state_dict(load_model.state_dict())
                ema_transformer.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = PixArtTransformer2DModel.from_pretrained(input_dir, subfolder="transformer")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_prodigy:
        try:
            from prodigyopt import Prodigy
        except ImportError:
            raise ImportError(
                "To use Prodigy optimizer, please install the prodigyopt library: `pip install prodigyopt`"
            )
        
        optimizer = Prodigy(
            transformer.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            d_coef=args.prodigy_d_coef,
            d0=args.prodigy_d0,
            safeguard_warmup=args.prodigy_safeguard_warmup,
            use_bias_correction=args.prodigy_use_bias_correction,
        )
        logger.info(f"Using Prodigy optimizer with d_coef={args.prodigy_d_coef}, d0={args.prodigy_d0}")
    elif args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
        optimizer = optimizer_cls(
            transformer.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    else:
        optimizer_cls = torch.optim.AdamW
        optimizer = optimizer_cls(
            transformer.parameters(),
            lr=args.learning_rate,
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
        )
        # We need to wrap it in a structure that allows "train" access if it's not a DatasetDict
        if not hasattr(dataset, "keys"):
             dataset = {"train": dataset}
    else:
        # Load custom RSCC Dataset
        train_dataset = RSCCDataset(
            data_path=args.train_data_dir,
            max_samples=args.max_train_samples
        )
        # Create a fake dataset dict for compatibility with later code (if we used it, but we'll override)
        dataset = {"train": train_dataset}

    # 6. Get the column names for input/target.
    # HARDCODED for RSCCDataset since we know what it returns
    original_image_column = "input_image"
    edit_prompt_column = "edit_prompt"
    edited_image_column = "edited_image"

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(captions):
        # T5 Tokenizer
        inputs = tokenizer(
            captions, max_length=120, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        ]
    )

    def convert_to_np(image, resolution):
        image = image.convert("RGB").resize((resolution, resolution))
        return np.array(image).transpose(2, 0, 1)

    # Custom Collate Function to handle the PyTorch Dataset items
    def custom_collate_fn(batch):
        original_images = [x[original_image_column] for x in batch]
        edited_images = [x[edited_image_column] for x in batch]
        captions = [x[edit_prompt_column] for x in batch]
        
        # Convert to numpy and resize
        original_images_np = [convert_to_np(img, args.resolution) for img in original_images]
        edited_images_np = [convert_to_np(img, args.resolution) for img in edited_images]
        
        original_stack = np.stack(original_images_np) # (B, 3, H, W)
        edited_stack = np.stack(edited_images_np)     # (B, 3, H, W)
        
        # Stack together for consistent transform
        combined = np.concatenate([original_stack, edited_stack], axis=0) # (2B, 3, H, W)
        
        combined_tensor = torch.tensor(combined).float()
        combined_tensor = 2 * (combined_tensor / 255) - 1
        
        # Apply transforms
        transformed_combined = train_transforms(combined_tensor)
        
        # Split back
        original_pixel_values, edited_pixel_values = torch.chunk(transformed_combined, 2, dim=0)
        
        # Tokenize captions
        input_ids = tokenize_captions(captions)
        
        return {
            "original_pixel_values": original_pixel_values,
            "edited_pixel_values": edited_pixel_values,
            "input_ids": input_ids,
        }

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        dataset["train"] if args.dataset_name else dataset["train"],
        shuffle=True,
        collate_fn=custom_collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_transformer.to(accelerator.device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("rsedit-dit", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset['train'])}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
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
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(transformer):
                # Convert images to latent space
                latents = vae.encode(batch["edited_pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning (T5)
                encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device), attention_mask=None)[0]

                # Get the additional image embedding for conditioning (Source Image)
                source_image_embeds = vae.encode(batch["original_pixel_values"].to(weight_dtype)).latent_dist.mode()
                source_image_embeds = source_image_embeds * vae.config.scaling_factor # Ensure scale match? DiTStage2 uses it.

                # Conditioning dropout
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(bsz, device=latents.device, generator=generator)
                    
                    # 1. Text Conditioning Dropout
                    # Drop text if p < 2 * prob
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    
                    # Create null conditioning (empty string)
                    null_tokenized = tokenize_captions([""] * bsz).to(accelerator.device)
                    null_conditioning = text_encoder(null_tokenized, attention_mask=None)[0]
                    
                    encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)
                    
                    # 2. Image Conditioning Dropout
                    # Drop image if p >= prob AND p < 3 * prob
                    # This ensures we cover: Text Only Drop, Both Drop, Image Only Drop
                    image_mask_dtype = source_image_embeds.dtype
                    image_drop_mask = (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype) * \
                                      (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                    
                    image_mask = (1 - image_drop_mask).reshape(bsz, 1, 1, 1)
                    source_image_embeds = source_image_embeds * image_mask

                # Token Concatenation Strategy
                # Concatenate source and target latents along width dimension
                concatenated_latents = torch.cat([source_image_embeds, noisy_latents], dim=3)
                
                # Forward pass
                # Note: PixArt-Alpha might expect 'added_cond_kwargs' for resolution.
                # If we use standard PixArt, we might need to assume 512x512 resolution buckets.
                # However, for this 'light' script, we rely on the transformer handling the shape.
                # If it fails, we might need to inject dummy added_cond_kwargs.
                
                model_pred = transformer(
                    concatenated_latents, 
                    timestep=timesteps, 
                    encoder_hidden_states=encoder_hidden_states, 
                    return_dict=False
                )[0]
                
                # Extract target portion (right half)
                # Input width was W*2, output width should be W*2.
                # We want prediction for the noisy part (second half).
                width = latents.shape[3]
                model_pred = model_pred[:, :, :, width:]

                # Split model prediction if it contains variance (PixArt outputs 8 channels)
                if model_pred.shape[1] == 2 * latents.shape[1]:
                    model_pred, _ = model_pred.chunk(2, dim=1)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Compute loss
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_transformer.step(transformer.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
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

                if args.validation_steps is not None and global_step % args.validation_steps == 0:
                    if accelerator.is_main_process:
                        if (
                            (args.val_image_url is not None and args.validation_prompt is not None)
                            or validation_items
                        ):
                            if args.use_ema:
                                # Store the Transformer parameters temporarily and load the EMA parameters to perform inference.
                                ema_transformer.store(transformer.parameters())
                                ema_transformer.copy_to(transformer.parameters())
                            
                            log_validation(
                                vae,
                                text_encoder,
                                tokenizer,
                                transformer,
                                noise_scheduler,
                                args,
                                accelerator,
                                generator,
                                validation_items=validation_items,
                            )

                            if args.use_ema:
                                # Switch back to the original Transformer parameters.
                                ema_transformer.restore(transformer.parameters())

                            torch.cuda.empty_cache()

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if (
                ((args.val_image_url is not None and args.validation_prompt is not None) or validation_items)
                and (epoch % args.validation_epochs == 0)
            ):
                if args.use_ema:
                    # Store the Transformer parameters temporarily and load the EMA parameters to perform inference.
                    ema_transformer.store(transformer.parameters())
                    ema_transformer.copy_to(transformer.parameters())
                
                log_validation(
                    vae,
                    text_encoder,
                    tokenizer,
                    transformer,
                    noise_scheduler,
                    args,
                    accelerator,
                    generator,
                    validation_items=validation_items,
                )

                if args.use_ema:
                    # Switch back to the original Transformer parameters.
                    ema_transformer.restore(transformer.parameters())

                torch.cuda.empty_cache()

    # Create the RSEdit pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.use_ema:
            ema_transformer.copy_to(transformer.parameters())

        # Load base PixArt components but save as RSEditDiTPipeline
        base_pipeline = PixArtAlphaPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=unwrap_model(text_encoder),
            vae=unwrap_model(vae),
            transformer=unwrap_model(transformer),
            tokenizer=tokenizer,
            revision=args.revision,
            variant=args.variant,
        )
        
        # Create RSEdit pipeline with trained components
        pipeline = RSEditDiTPipeline(
            vae=base_pipeline.vae,
            text_encoder=base_pipeline.text_encoder,
            tokenizer=base_pipeline.tokenizer,
            transformer=base_pipeline.transformer,
            scheduler=base_pipeline.scheduler,
        )
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )
        
        if (args.val_image_url is not None and args.validation_prompt is not None) or validation_items:
            log_validation(
                vae,
                text_encoder,
                tokenizer,
                transformer,
                noise_scheduler,
                args,
                accelerator,
                generator,
                validation_items=validation_items,
            )

    accelerator.end_training()


if __name__ == "__main__":
    # Example CLI usage:
    # export MODEL_NAME="/path/to/model"
    # export DATASET_DIR="/path/to/dataset"
    # export OUTPUT_DIR="/path/to/checkpoint"
    # export VAL_SET_PATH="/path/to/dataset"
    # export VAL_ANNOTATION_PATH="/path/to/dataset"
    # export PYTORCH_CUDNN_SDPA_ENABLED=0 
    # export CUDNN_FRONTEND_HEURISTIC_MODE_DISABLE=1
    # 
    # screen -S rsedit_dit
    # CUDA_VISIBLE_DEVICES=0,1 accelerate launch  --num_processes=2  --mixed_precision=bf16  train_dit.py \
    #   --pretrained_model_name_or_path=$MODEL_NAME \
    #   --train_data_dir=$DATASET_DIR \
    #   --resolution=512 \
    #   --train_batch_size=16 \
    #   --gradient_accumulation_steps=1 \
    #   --learning_rate=1e-5 \
    #   --num_train_epochs=5 \
    #   --output_dir=$OUTPUT_DIR \
    #   --mixed_precision=bf16 \
    #   --checkpointing_steps=10 \
    #   --validation_steps=10 \
    #   --val_set_path=$VAL_SET_PATH \
    #   --val_annotation_path=$VAL_ANNOTATION_PATH \
    #   --report_to="tensorboard" \
    #   --seed=42 \
    #   --conditioning_dropout_prob=0.05
    #
    # To temporarily exit the screen (detach): Ctrl+A then D
    # To re-enter the running screen session: screen -r rsedit_dit
    # To terminate the screen session completely: screen -S rsedit_dit -X quit
    #
    # tensorboard --logdir /path/to/checkpoint
    
    main()

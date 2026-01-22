#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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

"""
Train script to fine-tune Stable Diffusion 3 (MMDiT) for RSEdit using Token Concatenation.

We follow the same strategy as the PixArt/DiT trainer but swap in SD3 components
and FlowMatch scheduler. Source-image latents are concatenated with noisy target
latents along the width dimension to teach in-context editing.
"""

import argparse
import json
import logging
import math
import os
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import List

import accelerate
import datasets
import numpy as np
import PIL
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
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

import diffusers
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, StableDiffusion3Pipeline
from diffusers.models.transformers import SD3Transformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils.constants import DIFFUSERS_REQUEST_TIMEOUT
import requests
import PIL.ImageOps

from dataset import RSCCDataset
from pipeline_rsedit_sd3 import RSEditSD3Pipeline


if is_wandb_available():
    import wandb

logger = get_logger(__name__, log_level="INFO")

WANDB_TABLE_COL_NAMES = ["original_image", "edited_image", "edit_prompt"]


def log_validation(
    vae,
    text_encoder,
    text_encoder_2,
    text_encoder_3,
    tokenizer,
    tokenizer_2,
    tokenizer_3,
    transformer,
    scheduler,
    args,
    accelerator,
    generator,
    validation_items=None,
    guidance_scale=5.0,
):
    """Run validation using RSEditSD3Pipeline."""
    if validation_items is None:
        if args.val_image_url and args.validation_prompt:
            validation_items = [(args.val_image_url, args.validation_prompt)]
        else:
            return
    
    # Skip validation if T5 is not loaded (using cached embeddings mode)
    if text_encoder_3 is None:
        logger.warning("Skipping validation: T5 encoder not loaded (using cached embeddings mode)")
        logger.warning("To run validation, either disable --t5_cache_path or run validation separately")
        return

    logger.info(f"Running validation on {len(validation_items)} images... (CFG scale: {guidance_scale})")
    logger.info(f"Using seed: {args.seed} for reproducible validation")

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
    
    pipeline_scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler.config)

    pipeline = RSEditSD3Pipeline(
        transformer=unwrapped_transformer,
        scheduler=pipeline_scheduler,
        vae=unwrap_model(vae),
        text_encoder=unwrap_model(text_encoder),
        tokenizer=tokenizer,
        text_encoder_2=unwrap_model(text_encoder_2),
        tokenizer_2=tokenizer_2,
        text_encoder_3=unwrap_model(text_encoder_3),
        tokenizer_3=tokenizer_3,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

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

    image_idx = 0
    for image_path, prompt in validation_items:
        logger.info(f"Validating: {image_path} | Prompt: {prompt}")
        original_image = download_image(image_path)
        
        # Extract image name from path for saving
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        cudnn_enabled_backup = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        try:
            # Create a new generator with the same seed for reproducibility
            # Use CPU generator for better reproducibility across platforms
            val_generator = torch.Generator(device="cpu").manual_seed(args.seed)
            edited_images = pipeline(
                prompt=prompt,
                source_image=original_image,
                negative_prompt="",
                num_inference_steps=28,
                guidance_scale=guidance_scale,
                num_images_per_prompt=args.num_validation_images,
                height=args.resolution,
                width=args.resolution,
                generator=val_generator,
                max_sequence_length=120,
            ).images
        finally:
            torch.backends.cudnn.enabled = cudnn_enabled_backup

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
                wandb_table.add_data(wandb.Image(original_image), wandb.Image(edited_image), prompt)
            
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

    for tracker in accelerator.trackers:
        if tracker.name == "wandb" and wandb_table is not None:
            tracker.log({"validation": wandb_table})
    
    # Restore original attention processors for training
    unwrapped_transformer.set_attn_processor(original_attn_processors)


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for RSEdit SD3.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--train_data_dir", type=str, default=None)
    parser.add_argument("--original_image_column", type=str, default="input_image")
    parser.add_argument("--edited_image_column", type=str, default="edited_image")
    parser.add_argument("--edit_prompt_column", type=str, default="edit_prompt")
    parser.add_argument("--val_image_url", type=str, default=None)
    parser.add_argument("--validation_prompt", type=str, default=None)
    parser.add_argument("--num_validation_images", type=int, default=4)
    parser.add_argument("--validation_epochs", type=int, default=1)
    parser.add_argument("--validation_steps", type=int, default=None)
    parser.add_argument("--val_set_path", type=str, default=None)
    parser.add_argument("--val_annotation_path", type=str, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="rsedit-sd3-model")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--center_crop", action="store_true", default=False)
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help='["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]',
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--conditioning_dropout_prob", type=float, default=None)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--use_prodigy", action="store_true", help="Whether to use Prodigy optimizer.")
    parser.add_argument("--prodigy_d_coef", type=float, default=1.0, help="Prodigy D coefficient (default: 1.0).")
    parser.add_argument("--prodigy_d0", type=float, default=1e-5, help="Prodigy initial D estimate (default: 1e-5).")
    parser.add_argument("--prodigy_safeguard_warmup", action="store_true", help="Use safeguard warmup for Prodigy.")
    parser.add_argument("--prodigy_use_bias_correction", action="store_true", help="Use bias correction for Prodigy.")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--non_ema_revision", type=str, default=None)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--t5_cache_path", type=str, default=None,
                        help="Path to pre-computed T5 embeddings cache. If provided, T5 model is not loaded to save VRAM.")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


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
                except Exception:
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


def convert_to_np(image, resolution):
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)


def main():
    args = parse_args()

    validation_items = None
    if args.val_set_path and args.val_annotation_path:
        validation_items = load_validation_data(args.val_set_path, args.val_annotation_path)

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError("Use `hf auth login` instead of passing --hub_token with --report_to=wandb.")

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message="Use `--variant=non_ema` instead of specifying --non_ema_revision.",
        )

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Create generator for training (conditioning dropout)
    # Use device-specific generator for training performance
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

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

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision
    )
    tokenizer_3 = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_3", revision=args.revision
    )

    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    # Only load T5 if not using cached embeddings (saves significant VRAM)
    use_cached_t5 = args.t5_cache_path is not None
    if use_cached_t5:
        logger.info(f"Using pre-computed T5 embeddings from: {args.t5_cache_path}")
        logger.info("Skipping T5 model loading to save VRAM")
        text_encoder_3 = None
    else:
        text_encoder_3 = T5EncoderModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder_3", revision=args.revision, variant=args.variant
        )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.non_ema_revision
    )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    if text_encoder_3 is not None:
        text_encoder_3.requires_grad_(False)

    if args.use_ema:
        ema_transformer = EMAModel(transformer.parameters(), model_cls=SD3Transformer2DModel, model_config=transformer.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 may fail for training; update to >=0.0.17 if you see issues."
                )
            transformer.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Please install it or disable the flag.")

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_transformer.save_pretrained(os.path.join(output_dir, "transformer_ema"))

                for _, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "transformer"))
                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "transformer_ema"), SD3Transformer2DModel)
                ema_transformer.load_state_dict(load_model.state_dict())
                ema_transformer.to(accelerator.device)
                del load_model

            for _ in range(len(models)):
                model = models.pop()
                load_model = SD3Transformer2DModel.from_pretrained(input_dir, subfolder="transformer")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize optimizer
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
    else:
        optimizer_cls = torch.optim.AdamW
        optimizer = optimizer_cls(
            transformer.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # Dataset
    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
        if not hasattr(dataset, "keys"):
            dataset = {"train": dataset}
    else:
        train_dataset = RSCCDataset(
            data_path=args.train_data_dir,
            max_samples=args.max_train_samples,
            t5_cache_path=args.t5_cache_path,
        )
        dataset = {"train": train_dataset}
    
    # Load T5 null embedding for CFG if using cached embeddings
    t5_null_embedding = None
    if use_cached_t5 and hasattr(train_dataset, 'get_t5_null_embedding'):
        t5_null_embedding = train_dataset.get_t5_null_embedding()
        if t5_null_embedding is not None:
            logger.info(f"Loaded T5 null embedding from cache, shape: {t5_null_embedding.shape}")

    original_image_column = args.original_image_column
    edit_prompt_column = args.edit_prompt_column
    edited_image_column = args.edited_image_column

    train_transforms = transforms.Compose(
        [
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        ]
    )

    def convert_tensor(image):
        image = image.convert("RGB").resize((args.resolution, args.resolution))
        arr = np.array(image).transpose(2, 0, 1)
        tensor = torch.tensor(arr).float()
        tensor = 2 * (tensor / 255.0) - 1
        return tensor

    def custom_collate_fn(batch):
        original_images = [x[original_image_column] for x in batch]
        edited_images = [x[edited_image_column] for x in batch]
        captions = [x[edit_prompt_column] for x in batch]

        original_tensors = torch.stack([convert_tensor(img) for img in original_images])
        edited_tensors = torch.stack([convert_tensor(img) for img in edited_images])

        combined = torch.cat([original_tensors, edited_tensors], dim=0)
        combined = train_transforms(combined)
        original_pixel_values, edited_pixel_values = torch.chunk(combined, 2, dim=0)

        result = {
            "original_pixel_values": original_pixel_values,
            "edited_pixel_values": edited_pixel_values,
            "captions": captions,
        }
        
        # Include cached T5 embeddings if available
        if 't5_embedding' in batch[0]:
            t5_embeddings = torch.stack([x['t5_embedding'] for x in batch])
            result['t5_embeddings'] = t5_embeddings
        
        return result

    train_dataloader = torch.utils.data.DataLoader(
        dataset["train"],
        shuffle=True,
        collate_fn=custom_collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_transformer.to(accelerator.device)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    if text_encoder_3 is not None:
        text_encoder_3.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                "Length mismatch between prepared dataloader and scheduler expectations. LR scheduler may behave unexpectedly."
            )
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("rsedit-sd3", config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset['train'])}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0
    resume_step = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting fresh.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    def encode_prompts(captions: List[str], cached_t5_embeds: torch.Tensor = None):
        """
        Encode prompts using CLIP and T5 encoders.
        
        Args:
            captions: List of caption strings
            cached_t5_embeds: Pre-computed T5 embeddings [batch, seq_len, hidden_dim].
                              If provided, T5 model is not used.
        """
        # CLIP 1
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs_2 = tokenizer_2(
            captions,
            padding="max_length",
            max_length=tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            prompt_embeds_1 = text_encoder(text_inputs.input_ids.to(accelerator.device), output_hidden_states=True)
            pooled_1 = prompt_embeds_1[0]
            clip_embeds_1 = prompt_embeds_1.hidden_states[-2]

            prompt_embeds_2 = text_encoder_2(text_inputs_2.input_ids.to(accelerator.device), output_hidden_states=True)
            pooled_2 = prompt_embeds_2[0]
            clip_embeds_2 = prompt_embeds_2.hidden_states[-2]

            # Use cached T5 embeddings if provided, otherwise compute
            if cached_t5_embeds is not None:
                t5_embeds = cached_t5_embeds.to(accelerator.device, dtype=weight_dtype)
            else:
                t5_inputs = tokenizer_3(
                    captions,
                    padding="max_length",
                    max_length=120,
                    truncation=True,
                    return_tensors="pt",
                )
                t5_embeds = text_encoder_3(t5_inputs.input_ids.to(accelerator.device))[0]

        clip_prompt_embeds = torch.cat([clip_embeds_1, clip_embeds_2], dim=-1)
        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, t5_embeds.shape[-1] - clip_prompt_embeds.shape[-1])
        )
        prompt_embeds = torch.cat([clip_prompt_embeds, t5_embeds], dim=-2)
        pooled_prompt_embeds = torch.cat([pooled_1, pooled_2], dim=-1)
        return prompt_embeds, pooled_prompt_embeds

    # Training loop
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(transformer):
                latents = vae.encode(batch["edited_pixel_values"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # For flow matching, sample continuous timesteps t in [0, 1]
                # t=0 corresponds to clean data, t=1 corresponds to noise
                # We use a logit-normal distribution for better training dynamics (SD3 default)
                u = torch.randn(bsz, device=latents.device)
                # Shift and scale parameters (from SD3 paper defaults)
                u = torch.sigmoid(u)  # Transform to (0, 1)
                timesteps = u * 1000  # Scale to [0, 1000] for model timestep embedding
                
                # Flow matching interpolation: x_t = (1 - t) * x_0 + t * noise
                # where t is in [0, 1], so we use u directly
                t = u.view(-1, 1, 1, 1)  # Reshape for broadcasting
                noisy_latents = (1 - t) * latents + t * noise

                source_latents = vae.encode(
                    batch["original_pixel_values"].to(accelerator.device, dtype=weight_dtype)
                ).latent_dist.mode()
                source_latents = (source_latents - vae.config.shift_factor) * vae.config.scaling_factor

                # Use cached T5 embeddings if available
                cached_t5 = batch.get("t5_embeddings", None)
                encoder_hidden_states, pooled_prompt_embeds = encode_prompts(batch["captions"], cached_t5)

                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(bsz, device=latents.device, generator=generator)

                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)

                    # Use cached null T5 embedding for CFG if available
                    if t5_null_embedding is not None:
                        null_t5_batch = t5_null_embedding.unsqueeze(0).expand(bsz, -1, -1)
                        null_encoder_hidden_states, _ = encode_prompts([""] * bsz, null_t5_batch)
                    else:
                        null_encoder_hidden_states, _ = encode_prompts([""] * bsz)
                    encoder_hidden_states = torch.where(prompt_mask, null_encoder_hidden_states, encoder_hidden_states)

                    image_mask_dtype = source_latents.dtype
                    image_drop_mask = (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype) * (
                        random_p < 3 * args.conditioning_dropout_prob
                    ).to(image_mask_dtype)
                    image_mask = (1 - image_drop_mask).reshape(bsz, 1, 1, 1)
                    source_latents = source_latents * image_mask

                concatenated_latents = torch.cat([source_latents, noisy_latents], dim=3)

                model_pred = transformer(
                    hidden_states=concatenated_latents,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]

                width = latents.shape[3]
                model_pred = model_pred[:, :, :, width:]
                if model_pred.shape[1] == 2 * latents.shape[1]:
                    model_pred, _ = model_pred.chunk(2, dim=1)

                target = noise - latents
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_transformer.step(transformer.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                logger.info(
                                    f"{len(checkpoints)} checkpoints exist, removing {len(removing_checkpoints)}."
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
                        if (args.val_image_url and args.validation_prompt) or validation_items:
                            if args.use_ema:
                                ema_transformer.store(transformer.parameters())
                                ema_transformer.copy_to(transformer.parameters())

                            log_validation(
                                vae,
                                text_encoder,
                                text_encoder_2,
                                text_encoder_3,
                                tokenizer,
                                tokenizer_2,
                                tokenizer_3,
                                transformer,
                                noise_scheduler,
                                args,
                                accelerator,
                                generator,
                                validation_items=validation_items,
                            )

                            if args.use_ema:
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
                    ema_transformer.store(transformer.parameters())
                    ema_transformer.copy_to(transformer.parameters())

                log_validation(
                    vae,
                    text_encoder,
                    text_encoder_2,
                    text_encoder_3,
                    tokenizer,
                    tokenizer_2,
                    tokenizer_3,
                    transformer,
                    noise_scheduler,
                    args,
                    accelerator,
                    generator,
                    validation_items=validation_items,
                )

                if args.use_ema:
                    ema_transformer.restore(transformer.parameters())
                torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.use_ema:
            ema_transformer.copy_to(transformer.parameters())

        # When saving, we need all components including T5
        # If T5 wasn't loaded during training, load it now just for saving
        if text_encoder_3 is None:
            logger.info("Loading T5 encoder for final pipeline save...")
            text_encoder_3_for_save = T5EncoderModel.from_pretrained(
                args.pretrained_model_name_or_path, 
                subfolder="text_encoder_3", 
                revision=args.revision, 
                variant=args.variant
            )
        else:
            text_encoder_3_for_save = unwrap_model(text_encoder_3)

        base_pipeline = StableDiffusion3Pipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            transformer=unwrap_model(transformer),
            text_encoder=unwrap_model(text_encoder),
            text_encoder_2=unwrap_model(text_encoder_2),
            text_encoder_3=text_encoder_3_for_save,
            vae=unwrap_model(vae),
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            tokenizer_3=tokenizer_3,
            revision=args.revision,
            variant=args.variant,
        )

        pipeline = RSEditSD3Pipeline(
            transformer=base_pipeline.transformer,
            scheduler=base_pipeline.scheduler,
            vae=base_pipeline.vae,
            text_encoder=base_pipeline.text_encoder,
            tokenizer=base_pipeline.tokenizer,
            text_encoder_2=base_pipeline.text_encoder_2,
            tokenizer_2=base_pipeline.tokenizer_2,
            text_encoder_3=base_pipeline.text_encoder_3,
            tokenizer_3=base_pipeline.tokenizer_3,
        )
        pipeline.save_pretrained(args.output_dir)
        
        # Clean up T5 if we loaded it just for saving
        if text_encoder_3 is None:
            del text_encoder_3_for_save
            torch.cuda.empty_cache()

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
                text_encoder_2,
                text_encoder_3,
                tokenizer,
                tokenizer_2,
                tokenizer_3,
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
    # export T5_CACHE_PATH="/path/to/dataset"
    # export PYTORCH_CUDNN_SDPA_ENABLED=0 
    # export CUDNN_FRONTEND_HEURISTIC_MODE_DISABLE=1
    # 
    # Step 1: Pre-compute T5 embeddings (run once to save VRAM during training):
    # python precompute_t5_embeddings.py \
    #   --pretrained_model_name_or_path=$MODEL_NAME \
    #   --data_path=$DATASET_DIR \
    #   --cache_path=$T5_CACHE_PATH \
    #   --batch_size=32
    #
    # Step 2: Train with cached T5 embeddings (saves ~11GB VRAM):
    # screen -S rsedit_sd3
    # CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 --mixed_precision=bf16 train_sd3.py \
    #   --pretrained_model_name_or_path=$MODEL_NAME \
    #   --train_data_dir=$DATASET_DIR \
    #   --resolution=1024 \
    #   --train_batch_size=8 \
    #   --gradient_accumulation_steps=1 \
    #   --learning_rate=5e-5 \
    #   --num_train_epochs=5 \
    #   --output_dir=$OUTPUT_DIR \
    #   --mixed_precision=bf16 \
    #   --checkpointing_steps=1000 \
    #   --t5_cache_path=$T5_CACHE_PATH \
    #   --report_to="tensorboard" \
    #   --seed=42 \
    #   --conditioning_dropout_prob=0.05
    #
    # Note: When using --t5_cache_path, validation is skipped because T5 is not loaded.
    # To run validation, either omit --t5_cache_path or run validation separately after training.
    #
    # To temporarily exit the screen (detach): Ctrl+A then D
    # To re-enter the running screen session: screen -r rsedit_sd3
    # To terminate the screen session completely: screen -S rsedit_sd3 -X quit
    #
    # tensorboard --logdir /path/to/checkpoint
    
    main()



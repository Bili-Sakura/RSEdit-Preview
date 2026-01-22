#!/usr/bin/env python3
"""
RSEdit Benchmark Generation Script

This script benchmarks various image editing models on multiple datasets.
Supports:
- Datasets: RSCC, LevirCC, SECOND-CC-AUG
- Image Editing Models: FLUX.1-Kontext (including NVFP4 single-file version), RSEdit-UNet, MagicBrush, Qwen-Image-Edit, InstructPix2Pix
- Text-to-Image Models with SDEdit: DiffusionSat, Text2Earth

Configuration:
- Default config loaded from: config_example.yaml
- Override with: --config my_config.yaml
- Default: seed=42, steps=100, model-specific guidance scales

TODO:
- Fix compatibility with Hopper GPUs (H100/H200): Current implementation may fail 
  with "system not yet initialized" errors on Hopper architecture

Usage:
    # Basic usage with RSCC (uses config_example.yaml)
    python generate.py --model_name flux-kontext
    
    # Use LevirCC dataset
    python generate.py --model_name flux-kontext --dataset_type levircc
    
    # Use SECOND-CC-AUG dataset
    python generate.py --model_name flux-kontext --dataset_type second_cc_aug
    
    # Custom config file
    python generate.py --model_name diffusionsat --config my_config.yaml --dataset_type levircc
    
    # Custom model path
    python generate.py --model_path /path/to/custom/model --model_type instruct_pix2pix
    
    # Single GPU inference on a specific GPU (e.g., GPU 1)
    # Sets CUDA_VISIBLE_DEVICES=1, uses cuda:0 internally
    python generate.py --model_name flux-kontext --device cuda:1
    
    # Multi-GPU inference (model parallelism across GPUs 0,1,2,3)
    # Sets CUDA_VISIBLE_DEVICES=0,1,2,3, uses device_map='balanced'
    python generate.py --model_name flux-kontext --device cuda:0,1,2,3
    
    # Multi-GPU on specific GPUs (e.g., GPUs 1 and 2) - multiple formats supported
    # Format 1: Comma-separated with cuda: prefix
    python generate.py --model_name qwen-image-edit-lightning --device cuda:1,cuda:2 --batch_size 4
    # Format 2: Comma-separated GPU IDs only
    python generate.py --model_name qwen-image-edit-lightning --device 1,2 --batch_size 4
    # Format 3: Space-separated
    python generate.py --model_name qwen-image-edit-lightning --device cuda:1 cuda:2 --batch_size 4
    
    # Skip existing outputs and use custom batch size
    python generate.py --model_name flux-kontext --device 0,1 --skip_existing --batch_size 4
"""

import argparse
import os
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Callable
from enum import Enum

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml


# =============================================================================
# Reproducibility Utilities
# =============================================================================

def enable_full_determinism():
    """
    Enable full deterministic behavior for reproducibility.
    
    This function sets up PyTorch and CUDA for deterministic operations,
    which may impact performance but ensures reproducible results.
    
    Based on: https://huggingface.co/docs/diffusers/using-diffusers/reproducibility
    """
    # Set CUBLAS to use deterministic algorithms
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    
    # Disable cuDNN benchmarking (non-deterministic)
    torch.backends.cudnn.benchmark = False
    
    # Enable cuDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    
    # Disable TensorFloat32 for better precision and consistency
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    # Use deterministic algorithms where available
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    print("Full determinism enabled (may impact performance)")


# =============================================================================
# Model Registry - Defines all supported models and their configurations
# =============================================================================

class ModelType(str, Enum):
    """Enumeration of supported model types."""
    FLUX_KONTEXT = "flux_kontext"
    INSTRUCT_PIX2PIX = "instruct_pix2pix"
    MAGIC_BRUSH = "magic_brush"
    QWEN_IMAGE_EDIT = "qwen_image_edit"
    QWEN_IMAGE_EDIT_LIGHTNING = "qwen_image_edit_lightning"  # Qwen-Image-Edit with Lightning acceleration
    SD3_INSTRUCT_PIX2PIX = "sd3_instruct_pix2pix"  # SD3 InstructPix2Pix
    SDEDIT_SD = "sdedit_sd"  # SDEdit with Stable Diffusion-based models
    DIFFUSIONSAT = "diffusionsat"  # DiffusionSat with metadata-conditioned UNet
    TEXT2EARTH = "text2earth"  # Text2Earth with resolution-controlled generation
    RSEDIT_ABLATION = "rsedit_ablation"  # RSEdit ablation models with custom UNet
    RSEDIT_DIT = "rsedit_dit"  # RSEdit DiT model with token concatenation


@dataclass
class ModelConfig:
    """Minimal configuration for a model in the registry."""
    path: str
    model_type: ModelType
    torch_dtype: str = "bfloat16"  # "float16" or "bfloat16"
    # Model-specific optimal defaults (can be overridden via config.yaml)
    guidance_scale: float = 1
    image_guidance_scale: float = 1.5  # For InstructPix2Pix/MagicBrush
    strength: float = 0.75  # For SDEdit models
    num_inference_steps: Optional[int] = None  # Model-specific inference steps (None = use default)


# Pre-defined model registry - maps model names to their configurations
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    # Image Editing Models
    "flux-kontext": ModelConfig(
        path="/path/to/model",
        model_type=ModelType.FLUX_KONTEXT,
        torch_dtype="bfloat16",
        guidance_scale=3.5,
        # num_inference_steps=28,
    ),
    "flux-kontext-nvfp4": ModelConfig(
        path="/path/to/model",
        model_type=ModelType.FLUX_KONTEXT,
        torch_dtype="bfloat16",
        guidance_scale=3.5,
        # num_inference_steps=28,
    ),
    "instruct-pix2pix": ModelConfig(
        path="/path/to/model",
        model_type=ModelType.INSTRUCT_PIX2PIX,
        guidance_scale=7.5,
        image_guidance_scale=1.5,
    ),
    "magic-brush": ModelConfig(
        path="/path/to/model",
        model_type=ModelType.MAGIC_BRUSH,
        guidance_scale=7.5,
        image_guidance_scale=1.5,
    ),
    "qwen-image-edit": ModelConfig(
        path="/path/to/model",
        model_type=ModelType.QWEN_IMAGE_EDIT,
        torch_dtype="bfloat16",
        guidance_scale=3.5,
    ),
    "qwen-image-edit-lightning": ModelConfig(
        path="/path/to/model",
        model_type=ModelType.QWEN_IMAGE_EDIT_LIGHTNING,
        torch_dtype="bfloat16",
        guidance_scale=1.0,  # Lightning uses true_cfg_scale instead
        num_inference_steps=8,  # Lightning models use 8 steps as per specification
    ),
    "rsedit-unet": ModelConfig(
        path="/path/to/checkpoint",
        model_type=ModelType.INSTRUCT_PIX2PIX,
        torch_dtype="bfloat16",
        guidance_scale=7.5,
        image_guidance_scale=1.5,
    ),
    # RSEdit Ablation Models (text encoder ablation study)
    "rsedit-ablation-dgtrs": ModelConfig(
        path="/path/to/checkpoint",
        model_type=ModelType.RSEDIT_ABLATION,
        torch_dtype="bfloat16",
        guidance_scale=7.5,
        image_guidance_scale=1.5,
    ),
    "rsedit-ablation-git": ModelConfig(
        path="/path/to/checkpoint",
        model_type=ModelType.RSEDIT_ABLATION,
        torch_dtype="bfloat16",
        guidance_scale=7.5,
        image_guidance_scale=1.5,
    ),
    "rsedit-ablation-remote": ModelConfig(
        path="/path/to/checkpoint",
        model_type=ModelType.RSEDIT_ABLATION,
        torch_dtype="bfloat16",
        guidance_scale=7.5,
        image_guidance_scale=1.5,
    ),
    "rsedit-ablation-openai": ModelConfig(
        path="/path/to/checkpoint",
        model_type=ModelType.RSEDIT_ABLATION,
        torch_dtype="bfloat16",
        guidance_scale=7.5,
        image_guidance_scale=1.5,
    ),
    "sd3-ultraedit-freeform": ModelConfig(
        path="/path/to/model",
        model_type=ModelType.SD3_INSTRUCT_PIX2PIX,
        torch_dtype="bfloat16",
        guidance_scale=7.5,
        image_guidance_scale=1.5,
    ),
    # SDEdit Models (Text-to-Image with img2img)
    "diffusionsat": ModelConfig(
        path="/path/to/model",
        model_type=ModelType.DIFFUSIONSAT,
        guidance_scale=7.5,
        strength=0.75,
    ),
    "text2earth": ModelConfig(
        path="/path/to/model",
        model_type=ModelType.TEXT2EARTH,
        guidance_scale=4.0,
        strength=0.75,
    ),
    "sd21-base": ModelConfig(
        path="/path/to/model",
        model_type=ModelType.SDEDIT_SD,
        guidance_scale=7.5,
        strength=0.75,
    ),
    "sd15-base": ModelConfig(
        path="/path/to/model",
        model_type=ModelType.SDEDIT_SD,
        guidance_scale=7.5,
        strength=0.75,
    ),
    # RSEdit DiT Model (Token Concatenation)
    "rsedit-dit": ModelConfig(
        path="/path/to/checkpoint",
        model_type=ModelType.RSEDIT_DIT,
        torch_dtype="bfloat16",
        guidance_scale=4.5,
        image_guidance_scale=1.5,
    ),
}


# =============================================================================
# Inference Configuration
# =============================================================================

@dataclass
class InferenceConfig:
    """Unified inference hyperparameter configuration."""
    # Core parameters
    seed: int = 42
    batch_size: int = 1
    num_inference_steps: int = 100
    guidance_scale: float = 1
    
    # InstructPix2Pix / MagicBrush specific
    image_guidance_scale: float = 1.5
    
    # Advanced / Model specific
    true_cfg_scale: float = 1.0  # Used by some models (e.g. FLUX) instead of guidance_scale
    
    # SDEdit specific (for T2I models)
    strength: float = 0.75
    
    # Resolution
    resolution: int = 512
    
    # Advanced options
    negative_prompt: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Pipeline Loaders
# =============================================================================

def load_flux_kontext_pipeline(model_path: str, device: str, dtype: torch.dtype, device_map=None):
    """Load FLUX Kontext pipeline."""
    from diffusers import FluxKontextPipeline
    import os
    
    # Check if this is the NVFP4 single-file version
    # NVFP4 version uses from_single_file with a safetensors file
    if "NVFP4" in model_path or "nvfp4" in model_path.lower():
        # Use the standard safetensors filename
        ckpt_path = os.path.join(model_path, "flux1-kontext-dev-nvfp4.safetensors")
        # Use local config path to avoid HuggingFace Hub authentication
        config_repo = "/path/to/model"
        if device_map:
            pipe = FluxKontextPipeline.from_single_file(
                ckpt_path,
                config=config_repo,
                torch_dtype=dtype,
                device_map=device_map,
            )
        else:
            pipe = FluxKontextPipeline.from_single_file(
                ckpt_path,
                config=config_repo,
                torch_dtype=dtype,
            ).to(device)
    else:
        # Standard from_pretrained loading
        if device_map:
            pipe = FluxKontextPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device_map,
            )
        else:
            pipe = FluxKontextPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
            ).to(device)
    # pipe.set_progress_bar_config(disable=True)
    return pipe


def load_instruct_pix2pix_pipeline(model_path: str, device: str, dtype: torch.dtype, device_map=None):
    """Load InstructPix2Pix pipeline."""
    from diffusers import StableDiffusionInstructPix2PixPipeline
    
    if device_map:
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            safety_checker=None,
            device_map=device_map,
        )
    else:
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            safety_checker=None,
        ).to(device)
    # pipe.set_progress_bar_config(disable=True)
    return pipe


def load_magic_brush_pipeline(model_path: str, device: str, dtype: torch.dtype, device_map=None):
    """
    Load MagicBrush pipeline.
    MagicBrush uses the InstructPix2Pix architecture.
    """
    from diffusers import StableDiffusionInstructPix2PixPipeline
    
    if device_map:
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            safety_checker=None,
            # use_safetensors=False,
            device_map=device_map,
        )
    else:
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            safety_checker=None,
            # use_safetensors=False,
        ).to(device)
    # pipe.set_progress_bar_config(disable=True)
    return pipe


def load_rsedit_ablation_pipeline(checkpoint_path: str, device: str, dtype: torch.dtype, device_map=None):
    """
    Load RSEdit ablation model pipeline.
    
    Loads the base model but replaces the UNet with the trained UNet from checkpoint-30000/unet_ema.
    
    The checkpoint_path should point to the ablation experiment directory (e.g., 
    /path/to/checkpoint
    The function will automatically determine the base model and load checkpoint-30000/unet_ema.
    
    Args:
        checkpoint_path: Path to ablation experiment directory
        device: Device to load model on
        dtype: Torch dtype for model weights
        device_map: Optional device map for multi-GPU loading
    """
    from diffusers import StableDiffusionInstructPix2PixPipeline
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    import os
    
    # Map ablation experiment names to their base models (from ablation.sh)
    BASE_MODEL_MAP = {
        "DGTRS-CLIP-ViT-L-14": "/path/to/model"
        "Git-RSCLIP-ViT-L-16": "/path/to/model"
        "Remote-CLIP-ViT-L-14": "/path/to/model"
        "OPENAI-CLIP-VIT-L-14": "/path/to/model"
    }
    
    # Map ablation experiment names to their text encoders (from ablation.sh)
    TEXT_ENCODER_MAP = {
        "DGTRS-CLIP-ViT-L-14": "/path/to/model"
        "Git-RSCLIP-ViT-L-16": "/path/to/model"
        "Remote-CLIP-ViT-L-14": "/path/to/model"
        "OPENAI-CLIP-VIT-L-14": "/path/to/model"
    }
    
    # Extract experiment name from path
    experiment_name = os.path.basename(checkpoint_path.rstrip('/'))
    if experiment_name not in BASE_MODEL_MAP:
        raise ValueError(f"Unknown ablation experiment: {experiment_name}. Expected one of: {list(BASE_MODEL_MAP.keys())}")
    
    base_model_path = BASE_MODEL_MAP[experiment_name]
    text_encoder_path = TEXT_ENCODER_MAP[experiment_name]
    
    # Path to checkpoint-30000/unet_ema
    checkpoint_30000_dir = os.path.join(checkpoint_path, "checkpoint-30000")
    unet_ema_path = os.path.join(checkpoint_30000_dir, "unet_ema")
    
    if not os.path.exists(unet_ema_path):
        raise ValueError(f"UNet EMA checkpoint not found at {unet_ema_path}")
    
    # For multi-GPU, we need to handle component loading differently
    if device_map:
        # When using device_map, load components without moving to device
        vae = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae", torch_dtype=dtype)
        tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(text_encoder_path, torch_dtype=dtype)
        tokenizer.model_max_length = text_encoder.config.max_position_embeddings
        
        from diffusers import EulerDiscreteScheduler
        scheduler = EulerDiscreteScheduler.from_pretrained(base_model_path, subfolder="scheduler")
        
        unet = UNet2DConditionModel.from_pretrained(unet_ema_path, torch_dtype=dtype)
        
        # Create pipeline and let device_map handle placement
        pipe = StableDiffusionInstructPix2PixPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        
        # Apply device_map manually to each component
        if device_map in ["balanced", "auto"]:
            # Use accelerate for automatic device placement
            from accelerate import infer_auto_device_map, dispatch_model
            from accelerate.utils import get_balanced_memory
            
            # Get balanced memory based on available GPUs
            # We need to compute memory for individual components, not the pipeline
            # The UNet is typically the largest component, so we'll use it for memory calculation
            max_memory = get_balanced_memory(pipe.unet, dtype=dtype)
            device_map_unet = infer_auto_device_map(pipe.unet, max_memory=max_memory, dtype=dtype)
            pipe.unet = dispatch_model(pipe.unet, device_map=device_map_unet)
            
            # Move other components to first available device
            first_device = list(max_memory.keys())[0]
            pipe.vae = pipe.vae.to(first_device)
            pipe.text_encoder = pipe.text_encoder.to(first_device)
        else:
            # Custom device_map provided - apply it to each component
            pipe = pipe.to(device)
    else:
        # Single GPU or CPU - original loading logic
        vae = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae", torch_dtype=dtype)
        tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(text_encoder_path, torch_dtype=dtype)
        tokenizer.model_max_length = text_encoder.config.max_position_embeddings
        
        from diffusers import EulerDiscreteScheduler
        scheduler = EulerDiscreteScheduler.from_pretrained(base_model_path, subfolder="scheduler")
        
        unet = UNet2DConditionModel.from_pretrained(unet_ema_path, torch_dtype=dtype)
        
        pipe = StableDiffusionInstructPix2PixPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        
        pipe = pipe.to(device)
    
    # pipe.set_progress_bar_config(disable=True)
    return pipe


def load_qwen_image_edit_pipeline(model_path: str, device: str, dtype: torch.dtype, device_map=None):
    """
    Load Qwen Image Edit pipeline.
    
    CRITICAL: Disables cuDNN SDP backend to avoid cuDNN errors with batched inference.
    The cuDNN fused attention backend doesn't support certain graph patterns,
    especially with large batch sizes or non-standard sequence lengths.
    """
    # Disable cuDNN SDP backend to avoid "No execution plans support the graph" errors
    # This forces PyTorch to use alternative attention backends (Flash Attention or Math backend)
    torch.backends.cuda.enable_cudnn_sdp(False)
    
    from diffusers import QwenImageEditPipeline
    from diffusers.models import AutoModel
    
    if device_map:
        # Detect number of available GPUs (after CUDA_VISIBLE_DEVICES remapping)
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        # Create max_memory dict for A100 20GB GPUs
        # Use 20GB (full GPU) to ensure transformer loads on GPU instead of CPU
        max_memory = {i: "20GB" for i in range(num_gpus)} if num_gpus > 0 else None
        
        # Manually load transformer with device_map="auto" to ensure it's distributed across GPUs
        # This prevents the transformer from being offloaded to CPU
        print("Loading transformer with device_map='auto' for multi-GPU distribution...")
        transformer = AutoModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=dtype,
            device_map="auto",
            max_memory=max_memory,
        )
        
        # Debug: Print transformer device map
        if hasattr(transformer, 'hf_device_map'):
            print(f"\nTransformer device map:")
            for component, device_info in transformer.hf_device_map.items():
                print(f"  {component}: {device_info}")
        
        # Verify transformer is on GPU
        transformer_device = next(transformer.parameters()).device
        print(f"  Transformer first parameter device: {transformer_device}")
        if transformer_device.type == 'cpu':
            print(f"  WARNING: Transformer is still on CPU!")
        
        # Load pipeline with the manually loaded transformer
        # Use device_map="cuda" for other components to place them on GPU
        pipe = QwenImageEditPipeline.from_pretrained(
            model_path,
            transformer=transformer,
            torch_dtype=dtype,
            device_map="cuda",  # Place other components (VAE, text encoder) on GPU
        )
        
        # Debug: Print pipeline device map
        if hasattr(pipe, 'hf_device_map'):
            print(f"\nPipeline device map:")
            if isinstance(pipe.hf_device_map, dict):
                for component, device_info in pipe.hf_device_map.items():
                    print(f"  {component}: {device_info}")
            else:
                print(f"  {pipe.hf_device_map}")
    else:
        pipe = QwenImageEditPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
        ).to(device)
    # pipe.set_progress_bar_config(disable=True)
    return pipe


def load_qwen_image_edit_lightning_pipeline(model_path: str, device: str, dtype: torch.dtype, device_map=None):
    """
    Load Qwen Image Edit pipeline with Lightning acceleration.
    
    Uses FlowMatchEulerDiscreteScheduler with Lightning-specific config and loads
    Lightning LoRA weights for 8-step inference.
    
    CRITICAL: Disables cuDNN SDP backend to avoid cuDNN errors with batched inference.
    The cuDNN fused attention backend doesn't support certain graph patterns,
    especially with large batch sizes or non-standard sequence lengths.
    """
    # Disable cuDNN SDP backend to avoid "No execution plans support the graph" errors
    # This forces PyTorch to use alternative attention backends (Flash Attention or Math backend)
    torch.backends.cuda.enable_cudnn_sdp(False)
    
    import math
    from diffusers import QwenImageEditPipeline, FlowMatchEulerDiscreteScheduler
    
    # Lightning scheduler configuration (from Qwen-Image-Lightning README)
    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),  # We use shift=3 in distillation
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),  # We use shift=3 in distillation
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,  # set shift_terminal to None
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    
    # Load base model with Lightning scheduler
    if device_map:
        # Detect number of available GPUs (after CUDA_VISIBLE_DEVICES remapping)
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        # Create max_memory dict for A100 20GB GPUs
        # Use 20GB (full GPU) to ensure transformer loads on GPU instead of CPU
        max_memory = {i: "20GB" for i in range(num_gpus)} if num_gpus > 0 else None
        
        # Manually load transformer with device_map="auto" to ensure it's distributed across GPUs
        # This prevents the transformer from being offloaded to CPU
        print("Loading transformer with device_map='auto' for multi-GPU distribution...")
        from diffusers.models import AutoModel
        transformer = AutoModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=dtype,
            device_map="auto",
            max_memory=max_memory,
        )
        
        # Debug: Print transformer device map
        if hasattr(transformer, 'hf_device_map'):
            print(f"\nTransformer device map:")
            for component, device_info in transformer.hf_device_map.items():
                print(f"  {component}: {device_info}")
        
        # Verify transformer is on GPU
        transformer_device = next(transformer.parameters()).device
        print(f"  Transformer first parameter device: {transformer_device}")
        if transformer_device.type == 'cpu':
            print(f"  WARNING: Transformer is still on CPU!")
        
        # Load pipeline with the manually loaded transformer and Lightning scheduler
        # Use device_map="cuda" for other components to place them on GPU
        pipe = QwenImageEditPipeline.from_pretrained(
            model_path,
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=dtype,
            device_map="cuda",  # Place other components (VAE, text encoder) on GPU
        )
        
        # Debug: Print pipeline device map
        if hasattr(pipe, 'hf_device_map'):
            print(f"\nPipeline device map:")
            if isinstance(pipe.hf_device_map, dict):
                for component, device_info in pipe.hf_device_map.items():
                    print(f"  {component}: {device_info}")
            else:
                print(f"  {pipe.hf_device_map}")
    else:
        pipe = QwenImageEditPipeline.from_pretrained(
            model_path,
            scheduler=scheduler,
            torch_dtype=dtype,
        ).to(device)
    
    # Load Lightning LoRA weights
    # When using device_map, LoRA weights are automatically distributed across devices
    lightning_lora_dir = "/path/to/model"
    pipe.load_lora_weights(
        lightning_lora_dir,
        weight_name="Qwen-Image-Edit-Lightning-8steps-V1.0.safetensors"
    )
    
    # pipe.set_progress_bar_config(disable=True)
    return pipe


def load_ultraedit_pipeline(model_path: str, device: str, dtype: torch.dtype, device_map=None):
    """Load UltraEdit (SD3 InstructPix2Pix) pipeline."""
    # import sys
    # pipeline_path = "/path/to/libs/examples/community"
    # if pipeline_path not in sys.path:
    #     sys.path.insert(0, pipeline_path)
    
    from diffusers import StableDiffusion3InstructPix2PixPipeline
    
    if device_map:
        pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
        )
    else:
        pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
        ).to(device)
    return pipe


def load_sdedit_sd_pipeline(model_path: str, device: str, dtype: torch.dtype, device_map=None):
    """
    Load a Stable Diffusion pipeline for SDEdit.
    This works for generic SD-based T2I models.
    """
    from diffusers import StableDiffusionImg2ImgPipeline, AutoPipelineForImage2Image
    
    # Try AutoPipeline first, fall back to Img2Img
    try:
        if device_map:
            pipe = AutoPipelineForImage2Image.from_pretrained(
                model_path,
                torch_dtype=dtype,
                safety_checker=None,
                device_map=device_map,
            )
        else:
            pipe = AutoPipelineForImage2Image.from_pretrained(
                model_path,
                torch_dtype=dtype,
                safety_checker=None,
            ).to(device)
    except Exception:
        # Fallback: manually load components for Img2Img
        from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
        from transformers import CLIPTextModel, CLIPTokenizer
        
        # Load individual components
        vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype)
        unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=dtype)
        tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=dtype)
        scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
        
        pipe = StableDiffusionImg2ImgPipeline(
            vae=vae,
            unet=unet,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        
        if device_map:
            # For multi-GPU with manually constructed pipeline
            if device_map in ["balanced", "auto"]:
                from accelerate import infer_auto_device_map, dispatch_model
                from accelerate.utils import get_balanced_memory
                
                # Get balanced memory based on UNet (largest component)
                max_memory = get_balanced_memory(pipe.unet, dtype=dtype)
                device_map_unet = infer_auto_device_map(pipe.unet, max_memory=max_memory, dtype=dtype)
                pipe.unet = dispatch_model(pipe.unet, device_map=device_map_unet)
                
                first_device = list(max_memory.keys())[0]
                pipe.vae = pipe.vae.to(first_device)
                pipe.text_encoder = pipe.text_encoder.to(first_device)
            else:
                pipe = pipe.to(device)
        else:
            pipe = pipe.to(device)
    
    # pipe.set_progress_bar_config(disable=True)
    return pipe


def load_diffusionsat_pipeline(model_path: str, device: str, dtype: torch.dtype, device_map=None):
    """
    Load DiffusionSat pipeline with custom SatUNet.
    Uses the diffusionsat package's custom pipeline and UNet with metadata support.
    """
    # Add DiffusionSat to path if needed
    import sys
    diffusionsat_path = "/data/projects/DiffusionSat"
    if diffusionsat_path not in sys.path:
        sys.path.insert(0, diffusionsat_path)
    
    from diffusionsat import SatUNet, DiffusionSatPipeline
    from diffusers import AutoencoderKL, DDIMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    import os
    from safetensors import safe_open
    
    # Load components
    if device_map:
        # For multi-GPU, don't move components to device immediately
        vae = AutoencoderKL.from_pretrained(
            model_path, subfolder="vae", torch_dtype=dtype, low_cpu_mem_usage=False
        )
        text_encoder = CLIPTextModel.from_pretrained(
            model_path, subfolder="text_encoder", torch_dtype=dtype, low_cpu_mem_usage=False
        )
    else:
        # Single GPU - load and move to device
        vae = AutoencoderKL.from_pretrained(
            model_path, subfolder="vae", torch_dtype=dtype, low_cpu_mem_usage=False
        ).to(device)
        text_encoder = CLIPTextModel.from_pretrained(
            model_path, subfolder="text_encoder", torch_dtype=dtype, low_cpu_mem_usage=False
        ).to(device)
    
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
    
    # Load custom SatUNet with metadata support
    unet_path = os.path.join(model_path, "unet")
    unet_config = SatUNet.load_config(unet_path)
    unet = SatUNet(**unet_config)
    
    # Load weights manually
    weights_file = os.path.join(unet_path, "diffusion_pytorch_model.safetensors")
    if os.path.exists(weights_file):
        with safe_open(weights_file, framework='pt') as f:
            state_dict = {k: f.get_tensor(k) for k in f.keys()}
        unet.load_state_dict(state_dict, strict=False)
    else:
        # Fallback to .bin
        weights_file = os.path.join(unet_path, "diffusion_pytorch_model.bin")
        state_dict = torch.load(weights_file, map_location="cpu")
        unet.load_state_dict(state_dict, strict=False)
    
    if not device_map:
        unet = unet.to(device=device, dtype=dtype)
    else:
        unet = unet.to(dtype=dtype)
    
    # Create pipeline
    pipe = DiffusionSatPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    
    # Handle multi-GPU device placement
    if device_map:
        if device_map in ["balanced", "auto"]:
            from accelerate import infer_auto_device_map, dispatch_model
            from accelerate.utils import get_balanced_memory
            
            # Get balanced memory based on UNet (largest component)
            max_memory = get_balanced_memory(pipe.unet, dtype=dtype)
            device_map_unet = infer_auto_device_map(pipe.unet, max_memory=max_memory, dtype=dtype)
            pipe.unet = dispatch_model(pipe.unet, device_map=device_map_unet)
            
            first_device = list(max_memory.keys())[0]
            pipe.vae = pipe.vae.to(first_device)
            pipe.text_encoder = pipe.text_encoder.to(first_device)
        else:
            # Custom device_map - just use first available device
            pipe.vae = pipe.vae.to(device)
            pipe.text_encoder = pipe.text_encoder.to(device)
            pipe.unet = pipe.unet.to(device)
    
    # pipe.set_progress_bar_config(disable=True)
    return pipe


def load_text2earth_pipeline(model_path: str, device: str, dtype: torch.dtype, device_map=None):
    """
    Load Text2Earth pipeline with resolution control support.
    Uses the Text2Earth custom diffusers pipeline.
    """
    
    from diffusers import EulerDiscreteScheduler
    from diffusers import StableDiffusionPipeline
    
    # Load scheduler
    scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    
    # Load pipeline
    if device_map:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            scheduler=scheduler,
            safety_checker=None,
            custom_pipeline="pipeline_text2earth_diffusion",
            device_map=device_map,
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            scheduler=scheduler,
            safety_checker=None,
            custom_pipeline="pipeline_text2earth_diffusion"
        ).to(device)
    
    # pipe.set_progress_bar_config(disable=True)
    return pipe


def load_rsedit_dit_pipeline(model_path: str, device: str, dtype: torch.dtype, device_map=None):
    """
    Load RSEdit DiT pipeline with token concatenation strategy.
    
    Uses the RSEditDiTPipeline which extends PixArtAlphaPipeline to support
    image-to-image editing by concatenating source image tokens with noisy latent tokens.
    
    CRITICAL: Switches to AttnProcessor (non-SDPA) to avoid cuDNN errors with
    non-standard sequence lengths from token concatenation.
    """
    import sys
    # Add path to RSEdit pipeline module
    rsedit_light_path = "/path/to/src",
    if rsedit_light_path not in sys.path:
        sys.path.insert(0, rsedit_light_path)
    
    from pipeline_rsedit_dit import RSEditDiTPipeline
    
    # Load pipeline - RSEditDiTPipeline extends PixArtAlphaPipeline, so it supports from_pretrained
    if device_map:
        pipe = RSEditDiTPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
        )
    else:
        pipe = RSEditDiTPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
        ).to(device)
    
    # CRITICAL: Switch to AttnProcessor (non-SDPA) to avoid cuDNN errors
    # The default AttnProcessor2_0 uses F.scaled_dot_product_attention which
    # fails with cuDNN on non-standard sequence lengths from token concatenation
    from diffusers.models.attention_processor import AttnProcessor
    pipe.transformer.set_attn_processor(AttnProcessor())
    
    # pipe.set_progress_bar_config(disable=True)
    return pipe


# =============================================================================
# Inference Functions
# =============================================================================

def run_flux_kontext(pipe, images: List[Image.Image], prompts: List[str], config: InferenceConfig, generators: List[torch.Generator]):
    """
    Run inference with FLUX Kontext.
    
    Note: CLIP encoder will truncate prompts at 77 tokens (warning expected),
    but T5 encoder will process the full prompt up to 512 tokens.
    """
    result = pipe(
        image=images,
        prompt=prompts,
        guidance_scale=config.guidance_scale,
        num_inference_steps=config.num_inference_steps,
        generator=generators,
        height=config.resolution,
        width=config.resolution,
        max_sequence_length=512,  # Use full T5 capacity for long prompts
        _auto_resize=False,        # Keep your specified resolution
        max_area=config.resolution ** 2,  # Enforce resolution limit
        true_cfg_scale=config.true_cfg_scale,  # Pass true_cfg_scale
    )
    return result.images


def run_instruct_pix2pix(pipe, images: List[Image.Image], prompts: List[str], config: InferenceConfig, generators: List[torch.Generator]):
    """Run inference with InstructPix2Pix or MagicBrush."""
    result = pipe(
        prompt=prompts,
        image=images,
        num_inference_steps=config.num_inference_steps,
        image_guidance_scale=config.image_guidance_scale,
        guidance_scale=config.guidance_scale,
        generator=generators,
    )
    return result.images


def run_qwen_image_edit(pipe, images: List[Image.Image], prompts: List[str], config: InferenceConfig, generators: List[torch.Generator]):
    """Run inference with Qwen Image Edit."""
    result = pipe(
        image=images,
        prompt=prompts,
        guidance_scale=config.guidance_scale,
        num_inference_steps=config.num_inference_steps,
        generator=generators,
    )
    return result.images


def run_qwen_image_edit_lightning(pipe, images: List[Image.Image], prompts: List[str], config: InferenceConfig, generators: List[torch.Generator]):
    """
    Run inference with Qwen Image Edit Lightning acceleration.
    
    Uses 8 inference steps and true_cfg_scale=1.0 as per Lightning model specification.
    Based on: https://github.com/ModelTC/Qwen-Image-Lightning
    """
    # Lightning models use 8 steps by default (as per Lightning model specification)
    num_steps = 8
    
    # Use true_cfg_scale=1.0 as specified in Lightning README
    # If config has a different value > 1.0, use it; otherwise default to 1.0
    true_cfg_scale = config.true_cfg_scale if config.true_cfg_scale > 1.0 else 1.0
    
    # Only pass negative_prompt when CFG is enabled (true_cfg_scale > 1.0)
    # When CFG is disabled (true_cfg_scale <= 1.0), negative_prompt is ignored and causes a warning
    pipe_kwargs = {
        "image": images,
        "prompt": prompts,
        "num_inference_steps": num_steps,
        "true_cfg_scale": true_cfg_scale,
        "generator": generators,
    }
    
    # Only add negative_prompt if CFG is enabled
    if true_cfg_scale > 1.0:
        # Use empty string with space for negative_prompt to enable CFG (as per Lightning README)
        negative_prompt = config.negative_prompt if config.negative_prompt else " "
        pipe_kwargs["negative_prompt"] = negative_prompt
    
    result = pipe(**pipe_kwargs)
    return result.images


def run_sd3_instruct_pix2pix(pipe, images: List[Image.Image], prompts: List[str], config: InferenceConfig, generators: List[torch.Generator]):
    """Run inference with SD3 InstructPix2Pix."""
    result = pipe(
        prompt=prompts,
        image=images,
        height=config.resolution,
        width=config.resolution,
        guidance_scale=config.guidance_scale,
        image_guidance_scale=config.image_guidance_scale,
        num_inference_steps=config.num_inference_steps,
        generator=generators,
    )
    return result.images


def run_sdedit(pipe, images: List[Image.Image], prompts: List[str], config: InferenceConfig, generators: List[torch.Generator]):
    """Run SDEdit with a Stable Diffusion Img2Img pipeline."""
    result = pipe(
        prompt=prompts,
        image=images,
        strength=config.strength,
        guidance_scale=config.guidance_scale,
        num_inference_steps=config.num_inference_steps,
        generator=generators,
    )
    return result.images


@torch.no_grad()
def run_diffusionsat_sdedit(pipe, images: List[Image.Image], prompts: List[str], config: InferenceConfig, generators: List[torch.Generator]):
    """
    Run SDEdit with DiffusionSat pipeline (batched).
    
    DiffusionSat uses metadata conditioning. For SDEdit, we:
    1. Encode the input image to latents
    2. Add noise according to strength
    3. Denoise using DiffusionSat with default metadata (zeros)
    """
    device = pipe.device
    dtype = pipe.unet.dtype
    batch_size = len(images)
    
    # DiffusionSat needs CFG for proper generation - use at least 7.5
    guidance_scale = config.guidance_scale if config.guidance_scale > 1.0 else 7.5
    do_classifier_free_guidance = guidance_scale > 1.0
    
    # Prepare images
    from diffusers.image_processor import VaeImageProcessor
    image_processor = VaeImageProcessor(vae_scale_factor=pipe.vae_scale_factor)
    image_tensor = torch.cat([image_processor.preprocess(img) for img in images]).to(device=device, dtype=dtype)
    
    # Encode images to latents
    if isinstance(generators, list):
        latents = []
        for i in range(batch_size):
            l = pipe.vae.encode(image_tensor[i:i+1]).latent_dist.sample(generators[i])
            latents.append(l)
        latents = torch.cat(latents)
    else:
        latents = pipe.vae.encode(image_tensor).latent_dist.sample(generators)
    
    latents = latents * pipe.vae.config.scaling_factor
    
    # Set up timesteps based on strength
    pipe.scheduler.set_timesteps(config.num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    
    # Calculate starting step based on strength
    init_timestep = min(int(config.num_inference_steps * config.strength), config.num_inference_steps)
    t_start = max(config.num_inference_steps - init_timestep, 0)
    timesteps = timesteps[t_start:]
    
    # Add noise to latents
    if isinstance(generators, list):
        noise = []
        for i in range(batch_size):
            n = torch.randn(latents[i:i+1].shape, generator=generators[i], device=device, dtype=dtype)
            noise.append(n)
        noise = torch.cat(noise)
    else:
        noise = torch.randn(latents.shape, generator=generators, device=device, dtype=dtype)
        
    latents = pipe.scheduler.add_noise(latents, noise, timesteps[:1])
    
    # Encode prompt
    prompt_embeds = pipe._encode_prompt(
        prompts,
        device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_classifier_free_guidance,
        negative_prompt=config.negative_prompt,
    )
    
    # Prepare metadata (use zeros for default - no location conditioning)
    # DiffusionSat expects 7 metadata values: [lon, lat, gsd, cloud_cover, year, month, day]
    use_metadata = getattr(pipe.unet.config, 'use_metadata', False) or getattr(pipe.unet, 'use_metadata', False)
    if use_metadata:
        num_metadata = getattr(pipe.unet.config, 'num_metadata', 7) or getattr(pipe.unet, 'num_metadata', 7)
        metadata = torch.zeros(batch_size, num_metadata, device=device, dtype=dtype)
        if do_classifier_free_guidance:
            metadata = torch.cat([torch.zeros_like(metadata), metadata])
    else:
        metadata = None
    
    # Denoising loop
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generators[0] if isinstance(generators, list) else generators, eta=0.0)
    
    for i, t in enumerate(timesteps):
        # Expand latents for classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
        # Predict noise residual
        noise_pred = pipe.unet(
            latent_model_input,
            t,
            metadata=metadata,
            encoder_hidden_states=prompt_embeds,
        ).sample
        
        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Compute previous noisy sample
        latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
    
    # Decode latents - convert to float32 for VAE decode to avoid precision issues
    latents = 1 / pipe.vae.config.scaling_factor * latents
    latents = latents.to(torch.float32)
    
    # Temporarily set VAE to float32 for decoding
    original_dtype = pipe.vae.dtype
    pipe.vae = pipe.vae.to(torch.float32)
    image_output = pipe.vae.decode(latents).sample
    pipe.vae = pipe.vae.to(original_dtype)
    
    image_output = (image_output / 2 + 0.5).clamp(0, 1)
    
    # Convert to PIL - ensure proper data handling
    image_output = image_output.cpu().float().permute(0, 2, 3, 1).numpy()
    if np.isnan(image_output).any():
        # Replace NaN with 0
        image_output = np.nan_to_num(image_output, nan=0.0)
    image_output = (image_output * 255).round().clip(0, 255).astype(np.uint8)
    
    return [Image.fromarray(image_output[i]) for i in range(batch_size)]


@torch.no_grad()
def run_text2earth_sdedit(pipe, images: List[Image.Image], prompts: List[str], config: InferenceConfig, generators: List[torch.Generator]):
    """
    Run SDEdit with Text2Earth pipeline (batched).
    
    Text2Earth supports resolution control via prompt prefix.
    For SDEdit, we:
    1. Encode the input image to latents
    2. Add noise according to strength
    3. Denoise using Text2Earth
    
    Note: Text2Earth uses class_labels for resolution control (Google Map Level 10-18).
    Default level 17 corresponds to ~1m resolution.
    """
    device = pipe._execution_device
    dtype = pipe.unet.dtype
    batch_size = len(images)
    
    # Prepare images
    image_tensor = torch.cat([pipe.image_processor.preprocess(img) for img in images]).to(device=device, dtype=dtype)
    
    # Encode images to latents
    if isinstance(generators, list):
        latents = []
        for i in range(batch_size):
            l = pipe.vae.encode(image_tensor[i:i+1]).latent_dist.sample(generators[i])
            latents.append(l)
        latents = torch.cat(latents)
    else:
        latents = pipe.vae.encode(image_tensor).latent_dist.sample(generators)
    
    latents = latents * pipe.vae.config.scaling_factor
    
    # Set up timesteps based on strength
    pipe.scheduler.set_timesteps(config.num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    
    # Calculate starting step based on strength
    init_timestep = min(int(config.num_inference_steps * config.strength), config.num_inference_steps)
    t_start = max(config.num_inference_steps - init_timestep, 0)
    timesteps = timesteps[t_start:]
    
    # Add noise to latents
    if isinstance(generators, list):
        noise = []
        for i in range(batch_size):
            n = torch.randn(latents[i:i+1].shape, generator=generators[i], device=device, dtype=dtype)
            noise.append(n)
        noise = torch.cat(noise)
    else:
        noise = torch.randn(latents.shape, generator=generators, device=device, dtype=dtype)
        
    latents = pipe.scheduler.add_noise(latents, noise, timesteps[:1])
    
    do_classifier_free_guidance = config.guidance_scale > 1.0
    
    # Parse resolution from prompt if present, otherwise use default
    # Format: "{level}_GOOGLE_LEVEL_actual prompt"
    res_levels = []
    actual_prompts = []
    for prompt in prompts:
        if '_GOOGLE_LEVEL_' in prompt:
            res_levels.append(int(prompt.split('_GOOGLE_LEVEL_')[0]))
            actual_prompts.append(prompt.split('_GOOGLE_LEVEL_')[-1])
        else:
            res_levels.append(17)  # Default: ~1m resolution
            actual_prompts.append(prompt)
    
    # Encode prompt
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        actual_prompts,
        device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_classifier_free_guidance,
        negative_prompt=config.negative_prompt,
    )
    
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    
    # Prepare resolution conditioning
    res = torch.tensor(res_levels, dtype=torch.float32, device=device)
    res_null = torch.tensor([0] * batch_size, dtype=torch.float32, device=device)
    if do_classifier_free_guidance:
        res_in = torch.cat([res_null, res])
    else:
        res_in = res
    
    # Denoising loop
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generators[0] if isinstance(generators, list) else generators, eta=0.0)
    
    for i, t in enumerate(timesteps):
        # Expand latents for classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
        # Predict noise residual with resolution conditioning
        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            class_labels=res_in if pipe.unet.config.class_embed_type is not None else None,
            return_dict=False,
        )[0]
        
        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + config.guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Compute previous noisy sample
        latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
    
    # Decode latents
    image_outputs = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    
    # Post-process
    image_outputs = image_outputs.detach()
    pil_images = []
    for i in range(batch_size):
        img = pipe.image_processor.postprocess(image_outputs[i:i+1], output_type="pil")[0]
        pil_images.append(img)
    
    return pil_images


def run_rsedit_dit(pipe, images: List[Image.Image], prompts: List[str], config: InferenceConfig, generators: List[torch.Generator]):
    """
    Run inference with RSEdit DiT pipeline.
    
    RSEdit DiT uses token concatenation strategy where source image latents are
    concatenated with noisy target latents along the width dimension.
    """
    # RSEditDiTPipeline uses source_image parameter (not image)
    # Pass generators list directly - pipeline handles both single generator and list
    result = pipe(
        prompt=prompts,
        source_image=images,
        negative_prompt=config.negative_prompt if config.negative_prompt else "",
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        image_guidance_scale=config.image_guidance_scale if config.image_guidance_scale > 1.0 else None,
        height=config.resolution,
        width=config.resolution,
        generator=generators,
        clean_caption=False,
        max_sequence_length=120,
    )
    return result.images


# Model type to loader/runner mapping
MODEL_TYPE_CONFIG = {
    ModelType.FLUX_KONTEXT: {
        "loader": load_flux_kontext_pipeline,
        "runner": run_flux_kontext,
    },
    ModelType.INSTRUCT_PIX2PIX: {
        "loader": load_instruct_pix2pix_pipeline,
        "runner": run_instruct_pix2pix,
    },
    ModelType.MAGIC_BRUSH: {
        "loader": load_magic_brush_pipeline,
        "runner": run_instruct_pix2pix,  # Same interface as IP2P
    },
    ModelType.QWEN_IMAGE_EDIT: {
        "loader": load_qwen_image_edit_pipeline,
        "runner": run_qwen_image_edit,
    },
    ModelType.QWEN_IMAGE_EDIT_LIGHTNING: {
        "loader": load_qwen_image_edit_lightning_pipeline,
        "runner": run_qwen_image_edit_lightning,
    },
    ModelType.SD3_INSTRUCT_PIX2PIX: {
        "loader": load_ultraedit_pipeline,
        "runner": run_sd3_instruct_pix2pix,
    },
    ModelType.SDEDIT_SD: {
        "loader": load_sdedit_sd_pipeline,
        "runner": run_sdedit,
    },
    ModelType.DIFFUSIONSAT: {
        "loader": load_diffusionsat_pipeline,
        "runner": run_diffusionsat_sdedit,
    },
    ModelType.TEXT2EARTH: {
        "loader": load_text2earth_pipeline,
        "runner": run_text2earth_sdedit,
    },
    ModelType.RSEDIT_ABLATION: {
        "loader": load_rsedit_ablation_pipeline,
        "runner": run_instruct_pix2pix,  # Same interface as InstructPix2Pix
    },
    ModelType.RSEDIT_DIT: {
        "loader": load_rsedit_dit_pipeline,
        "runner": run_rsedit_dit,
    },
}


# =============================================================================
# Configuration Loading
# =============================================================================

def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """Load custom inference configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config if config else {}


# =============================================================================
# Dataset Loading
# =============================================================================

class DatasetType(str, Enum):
    """Enumeration of supported dataset types."""
    RSCC = "rscc"
    LEVIRCC = "levircc"
    SECOND_CC_AUG = "second_cc_aug"


def load_rscc_dataset(test_list_path: str, jsonl_path: str) -> List[Dict[str, Any]]:
    """Load RSCC test dataset."""
    print(f"Loading RSCC dataset from {test_list_path} and {jsonl_path}...")
    
    # Load test set paths
    with open(test_list_path, 'r') as f:
        test_paths = set(line.strip() for line in f if line.strip())
    
    # Load metadata and filter to test set
    samples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            pre_path = data['pre_image']
            if pre_path in test_paths:
                samples.append({
                    'pre_image': pre_path,
                    'post_image': data['post_image'],
                    'instruction': data['change_caption']
                })
    
    print(f"Found {len(samples)} samples matched in test set.")
    return samples


def load_levircc_dataset(json_path: str, base_dir: str = None) -> List[Dict[str, Any]]:
    """Load LevirCC test dataset."""
    if base_dir is None:
        base_dir = "datasets/LevirCC"
    
    print(f"Loading LevirCC dataset from {json_path}...")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    samples = []
    for img_info in data['images']:
        if img_info['split'] == 'test':
            filename = img_info['filename']
            filepath = img_info['filepath']
            
            # Pre image is in A directory, post image is in B directory
            pre_image = os.path.join(base_dir, filepath, "A", filename)
            post_image = os.path.join(base_dir, filepath, "B", filename)
            
            # Use the first caption as instruction (or combine all captions)
            if img_info['sentences']:
                # Use the first sentence as instruction
                instruction = img_info['sentences'][0]['raw'].strip()
            else:
                instruction = "Apply the change described in the image pair."
            
            samples.append({
                'pre_image': pre_image,
                'post_image': post_image,
                'instruction': instruction
            })
    
    print(f"Found {len(samples)} test samples.")
    return samples


def load_second_cc_aug_dataset(json_path: str, base_dir: str = None) -> List[Dict[str, Any]]:
    """Load SECOND-CC-AUG test dataset."""
    if base_dir is None:
        base_dir = "datasets/SECOND-CC-AUG"
    
    print(f"Loading SECOND-CC-AUG dataset from {json_path}...")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    samples = []
    for img_info in data['images']:
        if img_info['split'] == 'test':
            filename = img_info['filename']
            filepath = img_info['filepath']
            
            # Pre image is in rgb/A directory, post image is in rgb/B directory
            pre_image = os.path.join(base_dir, filepath, "rgb", "A", filename)
            post_image = os.path.join(base_dir, filepath, "rgb", "B", filename)
            
            # Use the first caption as instruction (or combine all captions)
            if img_info['sentences']:
                # Use the first sentence as instruction
                instruction = img_info['sentences'][0]['raw'].strip()
            else:
                instruction = "Apply the change described in the image pair."
            
            samples.append({
                'pre_image': pre_image,
                'post_image': post_image,
                'instruction': instruction
            })
    
    print(f"Found {len(samples)} test samples.")
    return samples


def load_dataset(dataset_type: str = None, test_list_path: str = None, jsonl_path: str = None, 
                 json_path: str = None, base_dir: str = None) -> List[Dict[str, Any]]:
    """
    Load dataset based on dataset type.
    
    Args:
        dataset_type: Type of dataset ('rscc', 'levircc', 'second_cc_aug')
        test_list_path: Path to test set list (for RSCC)
        jsonl_path: Path to JSONL metadata (for RSCC)
        json_path: Path to JSON metadata (for LevirCC/SECOND-CC-AUG)
        base_dir: Base directory for dataset (for LevirCC/SECOND-CC-AUG)
    
    Returns:
        List of samples with 'pre_image', 'post_image', and 'instruction' keys
    """
    if dataset_type is None:
        # Default to RSCC for backward compatibility
        dataset_type = DatasetType.RSCC.value
    
    if dataset_type == DatasetType.RSCC.value:
        if test_list_path is None or jsonl_path is None:
            raise ValueError("test_list_path and jsonl_path are required for RSCC dataset")
        return load_rscc_dataset(test_list_path, jsonl_path)
    elif dataset_type == DatasetType.LEVIRCC.value:
        if json_path is None:
            json_path = "/path/to/levircc",
        return load_levircc_dataset(json_path, base_dir)
    elif dataset_type == DatasetType.SECOND_CC_AUG.value:
        if json_path is None:
            json_path = "/path/to/second-cc-aug",
        return load_second_cc_aug_dataset(json_path, base_dir)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


# =============================================================================
# Main Generation Logic
# =============================================================================

def generate_filename(post_image_path: str, model_name: str) -> str:
    """
    Generate filename based on target image path with model name suffix.
    
    Args:
        post_image_path: Path to the target/post image
        model_name: Name of the model (will be sanitized for filename)
    
    Returns:
        Filename in format: {base_name}_{model_name}.png
    """
    # Extract base filename without extension
    base_name = os.path.splitext(os.path.basename(post_image_path))[0]
    
    # Sanitize model name for filename (replace spaces and dots with underscores)
    sanitized_model_name = model_name.replace(" ", "_").replace(".", "_")
    
    # Combine: base_name_{model_name}.png
    filename = f"{base_name}_{sanitized_model_name}.png"
    return filename


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate images using RSEdit baselines on RSCC test set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (loads config_example.yaml by default)
    python generate.py --model_name flux-kontext
    
    # Use custom config file
    python generate.py --model_name instruct-pix2pix --config my_config.yaml
    
    # Use a custom model path with specified type
    python generate.py --model_path /path/to/model --model_type instruct_pix2pix
    
    # Single GPU on GPU 1 (sets CUDA_VISIBLE_DEVICES=1)
    python generate.py --model_name flux-kontext --device cuda:1
    
    # Multi-GPU on GPUs 1 and 2 - easy formats:
    python generate.py --model_name qwen-image-edit-lightning --device 1,2 --batch_size 4
    python generate.py --model_name qwen-image-edit-lightning --device cuda:1,cuda:2 --batch_size 4
    python generate.py --model_name qwen-image-edit-lightning --device cuda:1 cuda:2 --batch_size 4
    
    # Multi-GPU with skip existing and custom batch size
    python generate.py --model_name flux-kontext --device 0,1,2,3 --skip_existing --batch_size 4
    
    # Test mode with verbose output (1 sample only)
    python generate.py --model_name diffusionsat --test

Available Models:
    """ + ", ".join(MODEL_REGISTRY.keys())
    )
    
    # Model selection
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model_name", type=str, choices=list(MODEL_REGISTRY.keys()),
        help="Name of pre-defined model from registry"
    )
    model_group.add_argument(
        "--model_path", type=str,
        help="Custom path to model (requires --model_type)"
    )
    
    parser.add_argument(
        "--model_type", type=str, choices=[t.value for t in ModelType],
        help="Model type (required when using --model_path)"
    )
    
    # Configuration
    parser.add_argument(
        "--config", type=str, 
        default=os.path.join(os.path.dirname(__file__), "config_example.yaml"),
        help="Path to YAML config file for inference parameters (default: config_example.yaml)"
    )
    
    # Dataset selection
    parser.add_argument(
        "--dataset_type", type=str,
        choices=[dt.value for dt in DatasetType],
        default=DatasetType.RSCC.value,
        help="Type of dataset to use (default: rscc)"
    )
    
    # Dataset paths (for RSCC)
    parser.add_argument(
        "--test_list", type=str, 
        default="/path/to/dataset",
        help="Path to test set list (for RSCC dataset)"
    )
    parser.add_argument(
        "--jsonl_path", type=str,
        default="/path/to/dataset",
        help="Path to RSCC jsonl metadata"
    )
    
    # Dataset paths (for LevirCC and SECOND-CC-AUG)
    parser.add_argument(
        "--json_path", type=str, default=None,
        help="Path to JSON metadata file (for LevirCC/SECOND-CC-AUG). Defaults based on dataset_type."
    )
    parser.add_argument(
        "--base_dir", type=str, default=None,
        help="Base directory for dataset (for LevirCC/SECOND-CC-AUG). Defaults based on dataset_type."
    )
    
    # Output configuration
    parser.add_argument(
        "--output_root", type=str, 
        default="/path/to/results",
        help="Root directory to save results"
    )
    parser.add_argument(
        "--output_name", type=str, default=None,
        help="Name of output directory (defaults to model name)"
    )
    
    # Device and GPU configuration
    parser.add_argument(
        "--device", type=str, nargs='*', default=None,
        help="Device(s) for inference. Supports multiple formats:\n"
             "  Single GPU: 'cuda:0', 'cuda:1', 'cpu'\n"
             "  Multi-GPU (comma-separated): 'cuda:1,cuda:2' or '1,2'\n"
             "  Multi-GPU (space-separated): 'cuda:1' 'cuda:2'\n"
             "Examples: --device cuda:1,2  or  --device cuda:1 cuda:2\n"
             "Default: cuda:0 if not specified"
    )
    parser.add_argument(
        "--gpus", type=str, default=None,
        help="[DEPRECATED] Use --device instead. Enable multi-GPU with comma-separated GPU IDs (e.g., '1,2'). "
             "This overrides --device if specified."
    )
    
    # Run configuration
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--test", action="store_true", help="Test mode: run only 1 sample with verbose output")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for testing")
    parser.add_argument("--skip_existing", action="store_true", help="Skip samples that already have outputs")
    parser.add_argument("--no-deterministic", action="store_false", dest="deterministic", 
                        help="Disable full determinism (faster, but less reproducible)")
    parser.add_argument("--task_index", type=int, default=None, help="Task index for parallel processing (0-based)")
    parser.add_argument("--num_tasks", type=int, default=None, help="Total number of parallel tasks")
    parser.set_defaults(deterministic=True)
    
    return parser.parse_args()


def save_config(model_name: str, model_config: ModelConfig, inference_config: InferenceConfig, output_dir: str, args, output_root: str = None, scheduler_name: Optional[str] = None):
    """Save generation configuration to JSON."""
    config = {
        "model": {
            "name": model_name,
            "path": model_config.path,
            "type": model_config.model_type.value,
            "torch_dtype": model_config.torch_dtype,
            "scheduler": scheduler_name,
        },
        "inference": inference_config.to_dict(),
        "dataset": {
            "type": args.dataset_type,
            "test_list": args.test_list if hasattr(args, 'test_list') else None,
            "jsonl_path": args.jsonl_path if hasattr(args, 'jsonl_path') else None,
            "json_path": args.json_path if hasattr(args, 'json_path') else None,
            "base_dir": args.base_dir if hasattr(args, 'base_dir') else None,
        },
        "output": {
            "root": output_root or (args.output_root if hasattr(args, 'output_root') else None),
            "directory": output_dir,
        },
        "run_info": {
            "timestamp": time.strftime("%Y%m%d-%H%M%S"),
            "device": args.device,
            "multi_gpu": args.gpus if hasattr(args, 'gpus') and args.gpus else None,
            "limit": args.limit if args.limit else "full",
            "deterministic": args.deterministic,
            "test_mode": args.test,
        }
    }
    
    config_path = os.path.join(output_dir, "generation_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")


def main():
    args = parse_args()
    
    # ==========================================================================
    # GPU Configuration - Set CUDA_VISIBLE_DEVICES FIRST before any CUDA ops
    # ==========================================================================
    # Supports flexible device specification:
    # 1. Single GPU: --device cuda:0 or --device cuda:1
    # 2. Multi-GPU: --device cuda:1,cuda:2 or --device cuda:1 cuda:2 or --device 1,2
    # 3. Legacy: --gpus 1,2 (deprecated, use --device instead)
    # ==========================================================================
    
    device_map = None
    
    # Parse device argument - can be None, a string, or list of strings
    device_input = args.device
    if device_input is None or (isinstance(device_input, list) and len(device_input) == 0):
        # No device specified, use default
        device_str = "cuda:0"
    elif isinstance(device_input, list):
        # Multiple devices provided as separate arguments: ['cuda:1', 'cuda:2']
        device_str = ','.join(device_input)
    elif isinstance(device_input, str):
        # Single string: 'cuda:0' or 'cuda:1,cuda:2' or '1,2'
        device_str = device_input
    else:
        # Fallback
        device_str = str(device_input)
    
    # Handle legacy --gpus argument (takes precedence)
    if args.gpus:
        device_str = args.gpus
        print("Warning: --gpus is deprecated. Use --device instead (e.g., --device 1,2)")
    
    # Parse device string to extract GPU IDs
    # Handle formats: "cuda:1,cuda:2", "1,2", "cuda:1", "cpu"
    gpu_ids = []
    if ',' in device_str:
        # Multi-device: parse comma-separated values
        parts = [p.strip() for p in device_str.split(',')]
        for part in parts:
            if part.startswith('cuda:'):
                gpu_id = part.split(':')[1]
                try:
                    gpu_ids.append(int(gpu_id))
                except ValueError:
                    raise ValueError(f"Invalid GPU ID in device specification: {part}")
            elif part.isdigit():
                gpu_ids.append(int(part))
            elif part.lower() == 'cpu':
                raise ValueError("Cannot mix CPU with GPU devices")
    else:
        # Single device
        if device_str.startswith('cuda:'):
            gpu_id = device_str.split(':')[1]
            try:
                gpu_ids.append(int(gpu_id))
            except ValueError:
                raise ValueError(f"Invalid GPU ID: {gpu_id}")
        elif device_str.isdigit():
            gpu_ids.append(int(device_str))
        elif device_str.lower() == 'cpu':
            gpu_ids = []  # CPU mode
    
    # IMPORTANT: Set CUDA_VISIBLE_DEVICES BEFORE any CUDA operations
    # This ensures that only specified GPUs are visible to PyTorch
    if len(gpu_ids) > 1:
        # Multi-GPU mode: distribute model across specified GPUs
        cuda_visible = ",".join(map(str, gpu_ids))
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible
        
        print(f"\n{'='*60}")
        print(f"MULTI-GPU MODE: Model parallelism across GPUs {gpu_ids}")
        print(f"{'='*60}")
        print(f"  CUDA_VISIBLE_DEVICES={cuda_visible}")
        print(f"  Physical GPUs {gpu_ids} are now visible as logical cuda:0-{len(gpu_ids)-1}")
        print(f"  Device map: balanced (distributed across GPUs {gpu_ids})")
        
        # Use balanced device_map - diffusers/accelerate will distribute across visible GPUs
        device_map = "balanced"
        args.device = "cuda:0"  # First visible GPU after CUDA_VISIBLE_DEVICES remapping
    elif len(gpu_ids) == 1:
        # Single GPU mode
        cuda_id = str(gpu_ids[0])
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
        # After CUDA_VISIBLE_DEVICES, the GPU becomes cuda:0
        args.device = "cuda:0"
        
        print(f"\n{'='*60}")
        print(f"SINGLE GPU MODE: Physical GPU {gpu_ids[0]} (visible as cuda:0)")
        print(f"{'='*60}")
        print(f"  CUDA_VISIBLE_DEVICES={cuda_id}")
    else:
        # CPU mode
        args.device = "cpu"
        print(f"\n{'='*60}")
        print(f"CPU MODE")
        print(f"{'='*60}")
    
    # Initialize and verify CUDA device (after CUDA_VISIBLE_DEVICES is set)
    if args.device.startswith("cuda"):
        if torch.cuda.is_available():
            print(f"CUDA initialized on {args.device}")
            print(f"  Available devices: {torch.cuda.device_count()}")
            print(f"  Current device: {torch.cuda.current_device()}")
            if device_map:
                print(f"  Device map: {device_map} (model distributed across visible GPUs)")
        else:
            print(f"Warning: CUDA not available, falling back to CPU")
            args.device = "cpu"
            device_map = None
    
    print()
    
    # Enable full determinism if requested
    if args.deterministic:
        enable_full_determinism()
    
    # Resolve model configuration
    if args.model_name:
        model_config = MODEL_REGISTRY[args.model_name]
    else:
        if args.model_type is None:
            print("Error: --model_type is required when using --model_path")
            sys.exit(1)
        model_config = ModelConfig(
            path=args.model_path,
            model_type=ModelType(args.model_type),
        )
    
    # Load config from YAML (default: config_example.yaml)
    custom_config = {}
    if args.config and os.path.exists(args.config):
        print(f"Loading config from: {args.config}")
        custom_config = load_config_from_yaml(args.config)
    elif args.config:
        print(f"Warning: Config file not found: {args.config}")
        print("Using model-specific defaults instead")
    
    # Build inference configuration with defaults and model-specific settings
    # Priority: model-specific defaults > custom config > InferenceConfig defaults
    # Exception: num_inference_steps uses model-specific default if set, otherwise custom config, otherwise 100
    # Resolution defaults: 512 for RSCC, 256 for LevirCC and SECOND-CC-AUG
    default_resolution = 512 if args.dataset_type == DatasetType.RSCC.value else 256
    inference_config = InferenceConfig(
        seed=custom_config.get('seed', 42),
        batch_size=custom_config.get('batch_size', args.batch_size),
        num_inference_steps=model_config.num_inference_steps if model_config.num_inference_steps is not None else custom_config.get('num_inference_steps', 100),
        guidance_scale=custom_config.get('guidance_scale', model_config.guidance_scale),
        image_guidance_scale=custom_config.get('image_guidance_scale', model_config.image_guidance_scale),
        true_cfg_scale=custom_config.get('true_cfg_scale', 1.0),
        strength=custom_config.get('strength', model_config.strength),
        resolution=custom_config.get('resolution', default_resolution),
        negative_prompt=custom_config.get('negative_prompt', None),
    )
    
    # Setup output directory
    # Automatically adjust output_root based on dataset type to avoid overwriting,
    # but honor a user-specified --output_root override.
    if args.output_root != "/path/to/results"
        output_root = args.output_root
    elif args.dataset_type == DatasetType.LEVIRCC.value:
        output_root = "/path/to/results"
    elif args.dataset_type == DatasetType.SECOND_CC_AUG.value:
        output_root = "/path/to/results"
    else:  # RSCC (default)
        output_root = args.output_root
    
    model_name = args.model_name or os.path.basename(args.model_path.rstrip('/'))
    output_name = args.output_name or model_name.replace(" ", "_").replace(".", "_")
    output_dir = os.path.join(output_root, output_name)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Handle test mode
    if args.test:
        print("\n" + "="*60)
        print("TEST MODE ENABLED - Running 1 sample with verbose output")
        print("="*60 + "\n")
        args.limit = 1
    
    # Load dataset
    samples = load_dataset(
        dataset_type=args.dataset_type,
        test_list_path=args.test_list,
        jsonl_path=args.jsonl_path,
        json_path=args.json_path,
        base_dir=args.base_dir
    )
    if args.limit:
        samples = samples[:args.limit]
        print(f"Limiting to {args.limit} samples.")
    
    # Filter samples by task index for parallel processing
    # Preserve original indices as (original_index, sample) tuples
    samples_with_indices = list(enumerate(samples))
    if args.task_index is not None and args.num_tasks is not None:
        filtered_samples_with_indices = [(i, sample) for i, sample in samples_with_indices if i % args.num_tasks == args.task_index]
        print(f"Task {args.task_index}/{args.num_tasks}: Processing {len(filtered_samples_with_indices)} out of {len(samples)} samples.")
        samples_with_indices = filtered_samples_with_indices
        samples = [sample for _, sample in samples_with_indices]
    elif args.task_index is not None or args.num_tasks is not None:
        print("Warning: Both --task_index and --num_tasks must be specified for parallel processing. Ignoring.")
    
    # Determine torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(model_config.torch_dtype, torch.float16)
    
    # Load pipeline
    print(f"Loading model: {model_name}")
    print(f"  Path: {model_config.path}")
    print(f"  Type: {model_config.model_type.value}")
    print(f"  Dtype: {model_config.torch_dtype}")
    if device_map:
        print(f"  Device Map: {device_map}")
    else:
        print(f"  Device: {args.device}")
    
    type_config = MODEL_TYPE_CONFIG[model_config.model_type]
    loader = type_config["loader"]
    runner = type_config["runner"]
    
    try:
        pipe = loader(model_config.path, args.device, torch_dtype, device_map=device_map)
    except Exception as e:
        print(f"Failed to load pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Log scheduler and save config
    scheduler_name = pipe.scheduler.__class__.__name__
    print(f"  Scheduler: {scheduler_name}")
    save_config(model_name, model_config, inference_config, output_dir, args, output_root=output_root, scheduler_name=scheduler_name)
    
    # Run generation
    print(f"Starting generation with {len(samples)} samples...")
    metadata = []
    metadata_path = os.path.join(output_dir, "metadata.json")
    
    # Filter samples if skip_existing is set
    # samples_with_indices already contains (original_index, sample) tuples from task_index filtering
    if args.skip_existing:
        to_process = []
        for i, sample in samples_with_indices:
            filename = generate_filename(sample['post_image'], model_name)
            save_path = os.path.join(images_dir, filename)
            if os.path.exists(save_path):
                metadata.append({
                    "id": i,
                    "filename": filename,
                    "pre_image": sample['pre_image'],
                    "post_image": sample['post_image'],
                    "instruction": sample['instruction'],
                    "status": "skipped"
                })
            else:
                to_process.append((i, sample))
        
        # Save initial metadata with skipped entries
        if metadata:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
    else:
        to_process = samples_with_indices

    batch_size = inference_config.batch_size
    print(f"Processing in batches of {batch_size}")
    
    for b_idx in tqdm(range(0, len(to_process), batch_size), desc="Generating"):
        batch = to_process[b_idx : b_idx + batch_size]
        batch_indices = [item[0] for item in batch]
        batch_samples = [item[1] for item in batch]
        
        try:
            # Load and preprocess images
            # For safety, detect each image's resolution and use it as target resolution
            pre_images = []
            instructions = []
            image_resolutions = []
            
            # Check if this is an RSEdit model (trained on 512px, should generate at >= 512)
            is_rsedit_model = model_config.model_type in (ModelType.RSEDIT_DIT, ModelType.RSEDIT_ABLATION) or \
                              (model_name and model_name.startswith("rsedit-"))
            
            for sample in batch_samples:
                pre_img = Image.open(sample['pre_image']).convert("RGB")
                # Detect image resolution (use width or height, assuming square images)
                # If not square, use the smaller dimension for safety
                img_width, img_height = pre_img.size
                detected_resolution = min(img_width, img_height)
                
                # For RSEdit models: use at least 512px (model trained on 512px)
                # If input is smaller than 512, generate at 512; otherwise use input resolution
                if is_rsedit_model:
                    target_resolution = max(detected_resolution, 512) if detected_resolution > 0 else 512
                else:
                    target_resolution = detected_resolution if detected_resolution > 0 else inference_config.resolution
                
                image_resolutions.append(target_resolution)
                pre_img = pre_img.resize((target_resolution, target_resolution))
                pre_images.append(pre_img)
                instructions.append(sample['instruction'])
            
            # Use the most common resolution in the batch, or first image's resolution if all differ
            # This ensures batch processing works correctly
            if len(set(image_resolutions)) == 1:
                batch_resolution = image_resolutions[0]
            else:
                # Use the most common resolution, or first if tie
                resolution_counts = Counter(image_resolutions)
                batch_resolution = resolution_counts.most_common(1)[0][0]
                if args.test:
                    print(f"Warning: Batch has mixed resolutions {set(image_resolutions)}, using {batch_resolution}")
            
            # Update inference config resolution for this batch
            batch_inference_config = InferenceConfig(**{**inference_config.to_dict(), 'resolution': batch_resolution})
            
            # Create generators for reproducibility
            # One generator per sample in the batch
            # When using device_map, generators should be on CPU or the execution device
            if device_map:
                # For multi-GPU pipelines, use CPU generators or the pipeline's execution device
                gen_device = getattr(pipe, '_execution_device', 'cpu') if hasattr(pipe, '_execution_device') else 'cpu'
                generators = [torch.Generator(device=gen_device).manual_seed(inference_config.seed) for _ in range(len(batch))]
            else:
                generators = [torch.Generator(device=args.device).manual_seed(inference_config.seed) for _ in range(len(batch))]
            
            # Run inference
            if args.test:
                print(f"\nProcessing batch of {len(batch)} samples...")
                if is_rsedit_model:
                    print(f"Using resolution: {batch_resolution} (RSEdit model: min 512px)")
                else:
                    print(f"Using resolution: {batch_resolution} (detected from images)")
                start_time = time.time()
                
            edited_images = runner(pipe, pre_images, instructions, batch_inference_config, generators)
            
            if args.test:
                elapsed = time.time() - start_time
                print(f"Batch inference completed in {elapsed:.2f}s\n")
            
            # Save outputs and update metadata
            for j, (i, sample) in enumerate(batch):
                filename = generate_filename(sample['post_image'], model_name)
                save_path = os.path.join(images_dir, filename)
                edited_images[j].save(save_path)
                
                metadata.append({
                    "id": i,
                    "filename": filename,
                    "pre_image": sample['pre_image'],
                    "post_image": sample['post_image'],
                    "instruction": sample['instruction'],
                    "status": "success"
                })
            
            # Update metadata.json after each batch
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            print(f"\nError processing batch starting at index {batch_indices[0]}: {e}")
            # In test mode or for serious debugging, show the traceback
            if args.test:
                import traceback
                traceback.print_exc()
                
            for i, sample in batch:
                metadata.append({
                    "id": i,
                    "filename": None,
                    "pre_image": sample['pre_image'],
                    "post_image": sample['post_image'],
                    "instruction": sample['instruction'],
                    "status": f"error: {str(e)}"
                })
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            continue
    
    # Summary
    success_count = sum(1 for m in metadata if m["status"] == "success")
    error_count = sum(1 for m in metadata if m["status"].startswith("error"))
    skip_count = sum(1 for m in metadata if m["status"] == "skipped")
    
    print(f"\nGeneration completed!")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"  Skipped: {skip_count}")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()

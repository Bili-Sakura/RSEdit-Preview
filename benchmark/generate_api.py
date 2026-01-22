#!/usr/bin/env python3
"""
RSEdit Benchmark Generation Script - OpenRouter API Version

This script benchmarks image editing models via OpenRouter API on RSCC validation set.
Supports:
- Dataset: RSCC validation set (val_set.txt)
- API Models: openai/gpt-5-image, google/gemini-3-pro-image-preview

Configuration:
- Default config loaded from: config_example.yaml
- Override with: --config my_config.yaml
- API key: Loads OPENROUTER_API_KEY from ./path/to/.env file (relative to script)
  (or from environment variable if .env file not found)

Usage:
    # Basic usage with OpenAI GPT-5 Image
    # (API key loaded from /path/to/.env)
    python generate_api.py --model_name openai/gpt-5-image
    
    # Use Gemini model
    python generate_api.py --model_name google/gemini-3-pro-image-preview
    
    # Custom config file
    python generate_api.py --model_name openai/gpt-5-image --config my_config.yaml
    
    # Test mode with verbose output (1 sample only)
    python generate_api.py --model_name openai/gpt-5-image --test
"""

import argparse
import os
import json
import sys
import time
import base64
import io
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from enum import Enum

import requests
import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml


# =============================================================================
# API Configuration
# =============================================================================

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


# =============================================================================
# Model Registry - Defines all supported API models
# =============================================================================

class ModelType(str, Enum):
    """Enumeration of supported API model types."""
    OPENROUTER_IMAGE = "openrouter_image"


@dataclass
class ModelConfig:
    """Configuration for an API model."""
    name: str  # Model name for OpenRouter API
    model_type: ModelType
    # API-specific configuration
    supports_aspect_ratio: bool = False  # Whether model supports aspect_ratio config
    supports_image_size: bool = False  # Whether model supports image_size config (Gemini only)
    supports_size: bool = False  # Whether model supports size config (e.g., GPT-5-image)
    default_aspect_ratio: Optional[str] = None  # Default aspect ratio (e.g., "1:1")
    default_image_size: Optional[str] = None  # Default image size (e.g., "1K", "2K", "4K")
    default_size: Optional[str] = None  # Default size (e.g., "512x512", "1024x1024")


# Pre-defined model registry - maps model names to their configurations
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "openai/gpt-5-image": ModelConfig(
        name="openai/gpt-5-image",
        model_type=ModelType.OPENROUTER_IMAGE,
        supports_aspect_ratio=False,
        supports_image_size=False,
        supports_size=True,  # GPT-5-image supports size parameter
        default_size="1024x1024",  # Default is 1024x1024
    ),
    "google/gemini-3-pro-image-preview": ModelConfig(
        name="google/gemini-3-pro-image-preview",
        model_type=ModelType.OPENROUTER_IMAGE,
        supports_aspect_ratio=True,
        supports_image_size=True,
        default_aspect_ratio="1:1",
        default_image_size="1K",
    ),
}


# =============================================================================
# Inference Configuration
# =============================================================================

@dataclass
class InferenceConfig:
    """Unified inference hyperparameter configuration."""
    # Core parameters
    seed: int = 42  # Note: API models may not support seed control
    batch_size: int = 1  # API requests are typically one at a time
    
    # Image configuration (for Gemini models)
    aspect_ratio: Optional[str] = None  # e.g., "1:1", "16:9"
    image_size: Optional[str] = None  # e.g., "1K", "2K", "4K" (Gemini only)
    size: Optional[str] = None  # e.g., "512x512", "1024x1024" (for GPT-5-image)
    
    # Resolution (for reference, actual resolution depends on model)
    resolution: int = 512
    
    # Advanced options
    negative_prompt: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# API Client Functions
# =============================================================================

def get_api_key() -> str:
    """
    Get OpenRouter API key from environment variable or .env file.
    
    Checks in order:
    1. Environment variable OPENROUTER_API_KEY
    2. .env file at ./path/to/.env (relative to script directory)
    """
    # First check environment variable
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        return api_key
    
    # Try loading from .env file (relative to script directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_file = os.path.join(script_dir, "..", ".env")
    env_file = os.path.normpath(env_file)
    if os.path.exists(env_file):
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    # Parse KEY=VALUE format
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        # Remove quotes if present
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                        
                        if key == "OPENROUTER_API_KEY":
                            return value
        except Exception as e:
            print(f"Warning: Could not read .env file: {e}")
    
    # If still not found, raise error
    raise ValueError(
        "OPENROUTER_API_KEY not found. "
        f"Set it as an environment variable or in {env_file} file"
    )


def call_openrouter_api(
    model_name: str,
    prompt: str,
    pre_image_path: Optional[str] = None,
    image_config: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Call OpenRouter API for image generation/editing.
    
    Args:
        model_name: Model name (e.g., "openai/gpt-5-image")
        prompt: Text prompt describing the image edit
        pre_image_path: Optional path to pre-image (for image editing)
        image_config: Optional image configuration (aspect_ratio, image_size)
        api_key: OpenRouter API key (if None, reads from env)
    
    Returns:
        API response as dictionary
    """
    if api_key is None:
        api_key = get_api_key()
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Build messages
    messages = []
    
    # If we have a pre-image, include it in the message for image editing
    if pre_image_path and os.path.exists(pre_image_path):
        with open(pre_image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Determine image MIME type
        ext = os.path.splitext(pre_image_path)[1].lower()
        mime_type = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp',
        }.get(ext, 'image/jpeg')
        
        # Create a clear image editing prompt
        edit_prompt = f"Edit this image according to the following instruction: {prompt}. Generate the edited version of the image."
        
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_data}"
                    }
                },
                {
                    "type": "text",
                    "text": edit_prompt
                }
            ]
        })
    else:
        # Text-only prompt (for text-to-image generation)
        messages.append({
            "role": "user",
            "content": prompt
        })
    
    # Build payload
    payload = {
        "model": model_name,
        "messages": messages,
        "modalities": ["image", "text"],
        "stream": False,
    }
    
    # Add image_config if provided (for Gemini models)
    if image_config:
        payload["image_config"] = image_config
    
    # Make API request
    try:
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=300  # 5 minute timeout for image generation
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"API request failed: {e}")


def extract_images_from_response(response: Dict[str, Any]) -> List[Image.Image]:
    """
    Extract images from OpenRouter API response.
    
    Args:
        response: API response dictionary
    
    Returns:
        List of PIL Images
    """
    images = []
    
    if "choices" not in response:
        return images
    
    for choice in response["choices"]:
        message = choice.get("message", {})
        
        # Check for images in the message
        if "images" in message:
            for image_obj in message["images"]:
                if "image_url" in image_obj:
                    image_url = image_obj["image_url"]
                    if "url" in image_url:
                        # Extract base64 data from data URL
                        data_url = image_url["url"]
                        if data_url.startswith("data:image"):
                            # Extract base64 part (after comma)
                            base64_data = data_url.split(",", 1)[1]
                            image_bytes = base64.b64decode(base64_data)
                            image = Image.open(io.BytesIO(image_bytes))
                            images.append(image.convert("RGB"))
        
        # Also check delta for streaming responses (though we use stream=False)
        if "delta" in choice:
            delta = choice["delta"]
            if "images" in delta:
                for image_obj in delta["images"]:
                    if "image_url" in image_obj:
                        image_url = image_obj["image_url"]
                        if "url" in image_url:
                            data_url = image_url["url"]
                            if data_url.startswith("data:image"):
                                base64_data = data_url.split(",", 1)[1]
                                image_bytes = base64.b64decode(base64_data)
                                image = Image.open(io.BytesIO(image_bytes))
                                images.append(image.convert("RGB"))
    
    return images


def run_api_inference(
    model_config: ModelConfig,
    pre_images: List[Image.Image],
    pre_image_paths: List[str],
    prompts: List[str],
    config: InferenceConfig,
    api_key: Optional[str] = None
) -> List[Image.Image]:
    """
    Run inference using OpenRouter API.
    
    Args:
        model_config: Model configuration
        pre_images: List of pre-images (PIL Images) - may not be used if API doesn't support
        pre_image_paths: List of paths to pre-images (for API)
        prompts: List of text prompts
        config: Inference configuration
        api_key: Optional API key (reads from env if None)
    
    Returns:
        List of edited PIL Images
    """
    if api_key is None:
        api_key = get_api_key()
    
    results = []
    
    # Build image_config if model supports it
    image_config = None
    if model_config.supports_aspect_ratio or model_config.supports_image_size or model_config.supports_size:
        image_config = {}
        if model_config.supports_aspect_ratio:
            image_config["aspect_ratio"] = config.aspect_ratio or model_config.default_aspect_ratio
        if model_config.supports_image_size:
            image_config["image_size"] = config.image_size or model_config.default_image_size
        if model_config.supports_size:
            # Use resolution from config to build size string (e.g., "512x512")
            if config.size:
                image_config["size"] = config.size
            elif config.resolution:
                image_config["size"] = f"{config.resolution}x{config.resolution}"
            else:
                image_config["size"] = model_config.default_size
    
    # Process each sample (API typically handles one at a time)
    for i, (pre_image_path, prompt) in enumerate(zip(pre_image_paths, prompts)):
        try:
            # Call API
            response = call_openrouter_api(
                model_name=model_config.name,
                prompt=prompt,
                pre_image_path=pre_image_path,
                image_config=image_config,
                api_key=api_key
            )
            
            # Extract images from response
            generated_images = extract_images_from_response(response)
            
            if generated_images:
                # Resize to requested resolution (API models may return different sizes)
                generated_image = generated_images[0]
                if generated_image.size != (config.resolution, config.resolution):
                    generated_image = generated_image.resize(
                        (config.resolution, config.resolution),
                        Image.Resampling.LANCZOS
                    )
                results.append(generated_image)
            else:
                # If no images in response, create a placeholder or raise error
                raise RuntimeError(f"No images found in API response for sample {i}")
                
        except Exception as e:
            print(f"Error generating image for sample {i}: {e}")
            # Create a placeholder black image as fallback
            placeholder = Image.new("RGB", (config.resolution, config.resolution), color="black")
            results.append(placeholder)
    
    return results


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


def load_rscc_dataset(val_list_path: str, jsonl_path: str) -> List[Dict[str, Any]]:
    """Load RSCC validation dataset."""
    print(f"Loading RSCC dataset from {val_list_path} and {jsonl_path}...")
    
    # Load validation set paths
    with open(val_list_path, 'r') as f:
        val_paths = set(line.strip() for line in f if line.strip())
    
    # Load metadata and filter to validation set
    samples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            pre_path = data['pre_image']
            if pre_path in val_paths:
                samples.append({
                    'pre_image': pre_path,
                    'post_image': data['post_image'],
                    'instruction': data['change_caption']
                })
    
    print(f"Found {len(samples)} samples matched in validation set.")
    return samples


def load_dataset(val_list_path: str = None, jsonl_path: str = None) -> List[Dict[str, Any]]:
    """
    Load RSCC validation dataset.
    
    Args:
        val_list_path: Path to validation set list (default: val_set.txt)
        jsonl_path: Path to JSONL metadata (default: RSCC_qvq.jsonl)
    
    Returns:
        List of samples with 'pre_image', 'post_image', and 'instruction' keys
    """
    if val_list_path is None:
        val_list_path = "/path/to/dataset"
    if jsonl_path is None:
        jsonl_path = "/path/to/dataset"
    
    return load_rscc_dataset(val_list_path, jsonl_path)


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
    
    # Sanitize model name for filename (replace spaces, dots, slashes with underscores)
    sanitized_model_name = model_name.replace(" ", "_").replace(".", "_").replace("/", "_")
    
    # Combine: base_name_{model_name}.png
    filename = f"{base_name}_{sanitized_model_name}.png"
    return filename


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate images using OpenRouter API models on RSCC validation set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (loads config_example.yaml by default)
    export OPENROUTER_API_KEY=your_api_key
    python generate_api.py --model_name openai/gpt-5-image
    
    # Use Gemini model
    python generate_api.py --model_name google/gemini-3-pro-image-preview
    
    # Custom config file
    python generate_api.py --model_name openai/gpt-5-image --config my_config.yaml
    
    # Test mode with verbose output (1 sample only)
    python generate_api.py --model_name openai/gpt-5-image --test

Available Models:
    """ + ", ".join(MODEL_REGISTRY.keys())
    )
    
    # Model selection
    parser.add_argument(
        "--model_name", type=str, choices=list(MODEL_REGISTRY.keys()),
        required=True,
        help="Name of pre-defined model from registry"
    )
    
    # Configuration
    parser.add_argument(
        "--config", type=str, 
        default=os.path.join(os.path.dirname(__file__), "config_example.yaml"),
        help="Path to YAML config file for inference parameters (default: config_example.yaml)"
    )
    
    # Dataset paths
    parser.add_argument(
        "--val_list", type=str, 
        default="/path/to/dataset"
        help="Path to validation set list (for RSCC dataset)"
    )
    parser.add_argument(
        "--jsonl_path", type=str,
        default="/path/to/dataset"
        help="Path to RSCC jsonl metadata"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_root", type=str, 
        default="/path/to/results"
        help="Root directory to save results"
    )
    parser.add_argument(
        "--output_name", type=str, default=None,
        help="Name of output directory (defaults to model name)"
    )
    
    # Run configuration
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (API processes one at a time)")
    parser.add_argument("--test", action="store_true", help="Test mode: run only 1 sample with verbose output")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for testing")
    parser.add_argument("--skip_existing", action="store_true", help="Skip samples that already have outputs")
    
    return parser.parse_args()


def save_config(model_name: str, model_config: ModelConfig, inference_config: InferenceConfig, output_dir: str, args, output_root: str = None):
    """Save generation configuration to JSON."""
    config = {
        "model": {
            "name": model_name,
            "type": model_config.model_type.value,
            "supports_aspect_ratio": model_config.supports_aspect_ratio,
            "supports_image_size": model_config.supports_image_size,
        },
        "inference": inference_config.to_dict(),
        "dataset": {
            "type": "rscc",
            "val_list": args.val_list,
            "jsonl_path": args.jsonl_path,
        },
        "output": {
            "root": output_root or args.output_root,
            "directory": output_dir,
        },
        "run_info": {
            "timestamp": time.strftime("%Y%m%d-%H%M%S"),
            "limit": args.limit if args.limit else "full",
            "test_mode": args.test,
        }
    }
    
    config_path = os.path.join(output_dir, "generation_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")


def main():
    args = parse_args()
    
    # Check for API key
    try:
        api_key = get_api_key()
        print("OpenRouter API key found.")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Resolve model configuration
    model_config = MODEL_REGISTRY[args.model_name]
    
    # Load config from YAML (default: config_example.yaml)
    custom_config = {}
    if args.config and os.path.exists(args.config):
        print(f"Loading config from: {args.config}")
        custom_config = load_config_from_yaml(args.config)
    elif args.config:
        print(f"Warning: Config file not found: {args.config}")
        print("Using model-specific defaults instead")
    
    # Build inference configuration with defaults and model-specific settings
    inference_config = InferenceConfig(
        seed=custom_config.get('seed', 42),
        batch_size=custom_config.get('batch_size', args.batch_size),
        aspect_ratio=custom_config.get('aspect_ratio', model_config.default_aspect_ratio),
        image_size=custom_config.get('image_size', model_config.default_image_size),
        size=custom_config.get('size', None),  # Will be set from resolution if not provided
        resolution=custom_config.get('resolution', 512),
        negative_prompt=custom_config.get('negative_prompt', None),
    )
    
    # Setup output directory
    output_root = args.output_root
    model_name = args.model_name
    output_name = args.output_name or model_name.replace(" ", "_").replace(".", "_").replace("/", "_")
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
        val_list_path=args.val_list,
        jsonl_path=args.jsonl_path
    )
    if args.limit:
        samples = samples[:args.limit]
        print(f"Limiting to {args.limit} samples.")
    
    # Log model info
    print(f"Using model: {model_name}")
    print(f"  Type: {model_config.model_type.value}")
    print(f"  Supports aspect_ratio: {model_config.supports_aspect_ratio}")
    print(f"  Supports image_size: {model_config.supports_image_size}")
    
    save_config(model_name, model_config, inference_config, output_dir, args, output_root=output_root)
    
    # Run generation
    print(f"Starting generation with {len(samples)} samples...")
    metadata = []
    metadata_path = os.path.join(output_dir, "metadata.json")
    
    # Filter samples if skip_existing is set
    samples_with_indices = list(enumerate(samples))
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

    print(f"Processing {len(to_process)} samples via API...")
    
    for i, sample in tqdm(to_process, desc="Generating"):
        try:
            # Load pre-image
            pre_image_path = sample['pre_image']
            pre_image = Image.open(pre_image_path).convert("RGB")
            pre_image = pre_image.resize((inference_config.resolution, inference_config.resolution))
            
            instruction = sample['instruction']
            
            # Run inference via API
            if args.test:
                print(f"\nProcessing sample {i}...")
                print(f"  Pre-image: {pre_image_path}")
                print(f"  Instruction: {instruction}")
                start_time = time.time()
            
            edited_images = run_api_inference(
                model_config=model_config,
                pre_images=[pre_image],
                pre_image_paths=[pre_image_path],
                prompts=[instruction],
                config=inference_config,
                api_key=api_key
            )
            
            if args.test:
                elapsed = time.time() - start_time
                print(f"API inference completed in {elapsed:.2f}s\n")
            
            # Save output and update metadata
            if edited_images:
                filename = generate_filename(sample['post_image'], model_name)
                save_path = os.path.join(images_dir, filename)
                edited_images[0].save(save_path)
                
                metadata.append({
                    "id": i,
                    "filename": filename,
                    "pre_image": sample['pre_image'],
                    "post_image": sample['post_image'],
                    "instruction": sample['instruction'],
                    "status": "success"
                })
            else:
                raise RuntimeError("No images returned from API")
            
            # Update metadata.json after each sample
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            print(f"\nError processing sample {i}: {e}")
            # In test mode, show the traceback
            if args.test:
                import traceback
                traceback.print_exc()
                
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
    # Basic usage
    # python generate_api.py --model_name openai/gpt-5-image

    # Use Gemini model
    # python generate_api.py --model_name google/gemini-3-pro-image-preview

    # Test mode
    # python generate_api.py --model_name openai/gpt-5-image --test
    main()

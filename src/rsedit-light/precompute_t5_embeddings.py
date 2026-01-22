#!/usr/bin/env python
# coding=utf-8
"""
Pre-compute T5 embeddings for RSEdit training.

This script pre-computes and caches T5 text embeddings to avoid loading
the large T5 model during training, significantly reducing VRAM usage.

Usage:
    python precompute_t5_embeddings.py \
        --pretrained_model_name_or_path /path/to/sd3-medium \
        --data_path /path/to/RSCC \
        --cache_path /path/to/RSCC/t5_embeddings_cache.pt \
        --batch_size 32
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import torch
from tqdm import tqdm
from transformers import T5EncoderModel, T5TokenizerFast


def load_annotations(data_path: str, annotation_file: str = "RSCC_qvq.jsonl") -> List[Dict[str, Any]]:
    """Load annotations from JSONL file."""
    annotation_path = os.path.join(data_path, annotation_file)
    samples = []
    
    if not os.path.exists(annotation_path):
        # Try alternate file names
        for alt_name in ["annotations.jsonl", "captions.jsonl", "train.jsonl"]:
            alt_path = os.path.join(data_path, alt_name)
            if os.path.exists(alt_path):
                annotation_path = alt_path
                break
        else:
            raise FileNotFoundError(f"Could not find annotation file in {data_path}")
    
    print(f"Loading annotations from: {annotation_path}")
    with open(annotation_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line.strip())
                    samples.append(item)
                except json.JSONDecodeError:
                    continue
    
    return samples


def get_unique_captions(samples: List[Dict[str, Any]]) -> Dict[str, int]:
    """Extract unique captions and map to indices."""
    caption_to_idx = {}
    for sample in samples:
        caption = sample.get('change_caption', 
                            sample.get('caption', 
                            sample.get('instruction', 'The scene has changed.')))
        if caption not in caption_to_idx:
            caption_to_idx[caption] = len(caption_to_idx)
    return caption_to_idx


def compute_t5_embeddings(
    captions: List[str],
    tokenizer: T5TokenizerFast,
    text_encoder: T5EncoderModel,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 120,
) -> torch.Tensor:
    """Compute T5 embeddings for a list of captions."""
    
    all_embeddings = []
    
    for i in tqdm(range(0, len(captions), batch_size), desc="Computing T5 embeddings"):
        batch_captions = captions[i:i + batch_size]
        
        inputs = tokenizer(
            batch_captions,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        with torch.no_grad():
            embeddings = text_encoder(inputs.input_ids.to(device))[0]
        
        all_embeddings.append(embeddings.cpu())
    
    return torch.cat(all_embeddings, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Pre-compute T5 embeddings for RSEdit")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                        help="Path to SD3 model (to load T5 from)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to RSCC dataset")
    parser.add_argument("--cache_path", type=str, default=None,
                        help="Path to save cache file (default: data_path/t5_embeddings_cache.pt)")
    parser.add_argument("--annotation_file", type=str, default="RSCC_qvq.jsonl",
                        help="Annotation file name")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for encoding")
    parser.add_argument("--max_length", type=int, default=120,
                        help="Max sequence length for T5")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--revision", type=str, default=None,
                        help="Model revision")
    parser.add_argument("--variant", type=str, default=None,
                        help="Model variant (e.g., fp16)")
    args = parser.parse_args()
    
    if args.cache_path is None:
        args.cache_path = os.path.join(args.data_path, "t5_embeddings_cache.pt")
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load annotations
    print("Loading annotations...")
    samples = load_annotations(args.data_path, args.annotation_file)
    print(f"Found {len(samples)} samples")
    
    # Get unique captions to avoid redundant computation
    caption_to_idx = get_unique_captions(samples)
    unique_captions = list(caption_to_idx.keys())
    print(f"Found {len(unique_captions)} unique captions")
    
    # Load T5 model
    print("Loading T5 encoder...")
    tokenizer = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="tokenizer_3",
        revision=args.revision
    )
    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_3",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    text_encoder = text_encoder.to(device)
    text_encoder.eval()
    
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        text_encoder = torch.nn.DataParallel(text_encoder)
    
    print("T5 encoder loaded")
    
    # Compute embeddings for unique captions
    print("Computing T5 embeddings...")
    unique_embeddings = compute_t5_embeddings(
        unique_captions,
        tokenizer,
        text_encoder,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print(f"Computed embeddings shape: {unique_embeddings.shape}")
    
    # Also compute null embedding for CFG
    print("Computing null embedding for CFG...")
    null_embedding = compute_t5_embeddings(
        [""],
        tokenizer,
        text_encoder,
        device,
        batch_size=1,
        max_length=args.max_length,
    )
    
    # Build sample index mapping (pre_image path -> embedding index)
    sample_to_embedding_idx = {}
    for sample in samples:
        pre_image = sample.get('pre_image', '')
        caption = sample.get('change_caption', 
                            sample.get('caption', 
                            sample.get('instruction', 'The scene has changed.')))
        sample_to_embedding_idx[pre_image] = caption_to_idx[caption]
    
    # Save cache
    cache_data = {
        'embeddings': unique_embeddings,  # [num_unique_captions, seq_len, hidden_dim]
        'null_embedding': null_embedding,  # [1, seq_len, hidden_dim]
        'caption_to_idx': caption_to_idx,  # {caption_str: idx}
        'sample_to_embedding_idx': sample_to_embedding_idx,  # {pre_image_path: idx}
        'max_length': args.max_length,
        'model_path': args.pretrained_model_name_or_path,
    }
    
    print(f"Saving cache to: {args.cache_path}")
    torch.save(cache_data, args.cache_path)
    
    # Print cache info
    cache_size_mb = os.path.getsize(args.cache_path) / (1024 * 1024)
    print(f"Cache saved successfully!")
    print(f"  - Cache size: {cache_size_mb:.2f} MB")
    print(f"  - Unique captions: {len(unique_captions)}")
    print(f"  - Embedding shape: {unique_embeddings.shape}")
    print(f"  - Total samples indexed: {len(sample_to_embedding_idx)}")
    
    # Cleanup
    del text_encoder
    torch.cuda.empty_cache()
    print("Done!")


if __name__ == "__main__":
    main()

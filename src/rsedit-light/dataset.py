"""
RSEdit Light Dataset Module
Modified from RSEdit/src/rsedit/data/dataset.py
"""

import os
import json
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
from typing import Optional, Callable, Dict, Any, List, Union
from pathlib import Path


class RSCCDataset(Dataset):
    """
    Remote Sensing Change Captioning Dataset.
    
    Loads data from RSCC_qvq.jsonl file which contains entries like:
    {
        "pre_image": "/path/to/pre.png",
        "post_image": "/path/to/post.png", 
        "change_caption": "Description of changes..."
    }
    
    Supports loading pre-computed T5 embeddings to reduce VRAM usage during training.
    """
    
    def __init__(
        self,
        data_path: str,
        transform: Optional[Callable] = None,
        annotation_file: str = "RSCC_qvq.jsonl",
        path_prefix: str = None,
        max_samples: int = None,
        exclude_test_set: bool = True,
        test_set_path: Optional[str] = None,
        t5_cache_path: Optional[str] = None,
    ):
        """
        Args:
            data_path: Root path to the RSCC dataset
            transform: Transform to apply to images
            annotation_file: Name of the JSONL annotation file
            path_prefix: Optional prefix to prepend to image paths (if paths are relative)
            max_samples: Maximum number of samples to load (for debugging)
            exclude_test_set: Whether to exclude test set images from training (default: True)
            test_set_path: Path to test_set.txt file. If None, looks for test_set.txt in data_path
            t5_cache_path: Path to pre-computed T5 embeddings cache file (optional)
        """
        self.data_path = data_path
        self.transform = transform
        self.path_prefix = path_prefix
        
        # Load T5 embeddings cache if provided
        self.t5_cache = None
        self.t5_embeddings = None
        self.t5_null_embedding = None
        self.t5_sample_to_idx = None
        self.t5_caption_to_idx = None
        
        if t5_cache_path is not None:
            self._load_t5_cache(t5_cache_path)
        
        # Load test set paths if excluding test set
        self.test_set_paths = set()
        if exclude_test_set:
            if test_set_path is None:
                test_set_path = os.path.join(data_path, "test_set.txt")
            if os.path.exists(test_set_path):
                with open(test_set_path, 'r', encoding='utf-8') as f:
                    self.test_set_paths = {line.strip() for line in f if line.strip()}
                print(f"Loaded {len(self.test_set_paths)} test set paths to exclude from training")
            else:
                print(f"Warning: test_set.txt not found at {test_set_path}. Test set exclusion disabled.")
        
        # Load annotations from JSONL
        annotation_path = os.path.join(data_path, annotation_file)
        if os.path.exists(annotation_path):
            self.samples = self._load_jsonl(annotation_path, max_samples)
        else:
            # Try alternate file names
            for alt_name in ["annotations.jsonl", "captions.jsonl", "train.jsonl", 
                             "RSCC_qvq_backup.jsonl"]:
                alt_path = os.path.join(data_path, alt_name)
                if os.path.exists(alt_path):
                    self.samples = self._load_jsonl(alt_path, max_samples)
                    break
            else:
                raise FileNotFoundError(
                    f"Could not find annotation file in {data_path}. "
                    f"Expected: {annotation_file}"
                )
        
        # Filter out test set samples
        if self.test_set_paths:
            original_count = len(self.samples)
            self.samples = [
                s for s in self.samples 
                if s.get('pre_image', '') not in self.test_set_paths
            ]
            excluded_count = original_count - len(self.samples)
            if excluded_count > 0:
                print(f"Excluded {excluded_count} test set samples from training")
        
        # Filter out samples with missing images
        self.samples = self._filter_valid_samples()
        
        print(f"Loaded {len(self.samples)} valid samples from RSCC dataset")
        
        if self.t5_cache is not None:
            print(f"T5 embeddings cache loaded: {len(self.t5_caption_to_idx)} unique captions")
    
    def _load_t5_cache(self, cache_path: str):
        """Load pre-computed T5 embeddings from cache file."""
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"T5 cache file not found: {cache_path}")
        
        print(f"Loading T5 embeddings cache from: {cache_path}")
        self.t5_cache = torch.load(cache_path, map_location='cpu')
        self.t5_embeddings = self.t5_cache['embeddings']
        self.t5_null_embedding = self.t5_cache['null_embedding']
        self.t5_sample_to_idx = self.t5_cache['sample_to_embedding_idx']
        self.t5_caption_to_idx = self.t5_cache['caption_to_idx']
        print(f"T5 embeddings shape: {self.t5_embeddings.shape}")
    
    def get_t5_embedding(self, caption: str) -> Optional[torch.Tensor]:
        """Get cached T5 embedding for a caption."""
        if self.t5_cache is None:
            return None
        
        if caption in self.t5_caption_to_idx:
            idx = self.t5_caption_to_idx[caption]
            return self.t5_embeddings[idx]
        return None
    
    def get_t5_null_embedding(self) -> Optional[torch.Tensor]:
        """Get cached null T5 embedding for CFG."""
        if self.t5_cache is None:
            return None
        return self.t5_null_embedding[0]  # Remove batch dimension
    
    def has_t5_cache(self) -> bool:
        """Check if T5 cache is loaded."""
        return self.t5_cache is not None
    
    def _load_jsonl(self, path: str, max_samples: int = None) -> List[Dict[str, Any]]:
        """Load annotations from JSONL file."""
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                if line.strip():
                    try:
                        item = json.loads(line.strip())
                        samples.append(item)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse line {i}: {e}")
                        continue
        return samples
    
    def _resolve_path(self, path: str) -> str:
        """Resolve image path, handling absolute and relative paths."""
        if os.path.isabs(path) and os.path.exists(path):
            return path
        
        # If path_prefix is provided, try prepending it
        if self.path_prefix:
            prefixed_path = os.path.join(self.path_prefix, path)
            if os.path.exists(prefixed_path):
                return prefixed_path
        
        # Try relative to data_path
        relative_path = os.path.join(self.data_path, path)
        if os.path.exists(relative_path):
            return relative_path
        
        # Try extracting just the filename and searching
        filename = os.path.basename(path)
        # Extract event name from the path
        path_parts = Path(path).parts
        for i, part in enumerate(path_parts):
            if 'images-w512-h512' in part or part in ['xbd', 'EBD']:
                # Reconstruct relative path from this point
                rel_path = os.path.join(self.data_path, *path_parts[i:])
                if os.path.exists(rel_path):
                    return rel_path
        
        return path  # Return original path, will fail later if not found
    
    def _filter_valid_samples(self) -> List[Dict[str, Any]]:
        """Filter samples to only include those with existing images."""
        valid_samples = []
        for sample in self.samples:
            pre_path = self._resolve_path(sample.get('pre_image', ''))
            post_path = self._resolve_path(sample.get('post_image', ''))
            
            if os.path.exists(pre_path) and os.path.exists(post_path):
                # Store resolved paths
                sample['_resolved_pre'] = pre_path
                sample['_resolved_post'] = post_path
                valid_samples.append(sample)
        
        return valid_samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Use resolved paths
        pre_path = sample.get('_resolved_pre', self._resolve_path(sample['pre_image']))
        post_path = sample.get('_resolved_post', self._resolve_path(sample['post_image']))
        
        # Load images
        pre_image = Image.open(pre_path).convert('RGB')
        post_image = Image.open(post_path).convert('RGB')
        
        # Get caption - RSCC uses 'change_caption' key
        caption = sample.get('change_caption', 
                            sample.get('caption', 
                            sample.get('instruction', 'The scene has changed.')))
        
        result = {
            'input_image': pre_image,
            'edited_image': post_image,
            'edit_prompt': caption,
        }
        
        # Add cached T5 embedding if available
        if self.t5_cache is not None:
            t5_embedding = self.get_t5_embedding(caption)
            if t5_embedding is not None:
                result['t5_embedding'] = t5_embedding
        
        if self.transform:
            # Note: transform usually handles only one image, here we need to apply consistent transform?
            # For now, let's assume transform is None here and handled in train loop
            pass
        
        return result

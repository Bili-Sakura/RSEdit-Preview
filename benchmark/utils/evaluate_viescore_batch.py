#!/usr/bin/env python3
"""
Batch VIEScore evaluation script for RSEdit benchmark
Evaluates all result folders under specified base directories using VIEScore metrics only.
"""

import argparse
import os
import json
import time
import random
import concurrent.futures
import threading
import sys
from PIL import Image
from tqdm import tqdm

# Add parent directory to sys.path to import viescore_openrouter
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from viescore_openrouter import VIEScoreEvaluator


# Retry configuration
MAX_RETRIES = 5
BASE_DELAY = 1.0  # seconds
MAX_DELAY = 60.0  # seconds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch evaluate all result folders with VIEScore"
    )
    parser.add_argument(
        "--base_dirs",
        type=str,
        nargs="+",
        default=[
            # "/path/to/results"
            # "/path/to/results"
            # "/path/to/results"
            "/path/to/results"
        ],
        help="Base directories containing result folders"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for VIEScore API calls (default: 8)"
    )
    parser.add_argument(
        "--viescore_model",
        type=str,
        default="google/gemini-3-flash-preview",
        help="Model name for VIEScore (e.g., google/gemini-3-flash-preview for OpenRouter, qwen-vl-max for DashScope)"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip result folders that already have viescore_metrics.json"
    )
    return parser.parse_args()


def find_result_folders(base_dirs):
    """Find all result folders with metadata.json in base directories."""
    result_folders = []
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"Warning: Base directory {base_dir} does not exist. Skipping.")
            continue
        
        # Allow passing a result folder directly
        metadata_path = os.path.join(base_dir, "metadata.json")
        if os.path.exists(metadata_path):
            result_folders.append(base_dir)
            print(f"  Found: {os.path.basename(base_dir)}")
            continue
        
        print(f"Scanning {base_dir}...")
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                metadata_path = os.path.join(item_path, "metadata.json")
                if os.path.exists(metadata_path):
                    result_folders.append(item_path)
                    print(f"  Found: {item}")
    
    return result_folders


def evaluate_folder(result_dir, evaluator, batch_size):
    """Evaluate a single result folder with VIEScore."""
    print(f"\n{'='*80}")
    print(f"Evaluating: {result_dir}")
    print(f"{'='*80}")
    
    metadata_path = os.path.join(result_dir, "metadata.json")
    with open(metadata_path, 'r') as f:
        samples = json.load(f)
    
    print(f"Found {len(samples)} samples to evaluate.")
    
    images_dir = os.path.join(result_dir, "images")
    
    # Load all images first
    print("Loading images...")
    loaded_samples = []
    for sample in tqdm(samples, desc="Loading"):
        try:
            # Load Edited Image
            edited_path = os.path.join(images_dir, sample['filename'])
            if not os.path.exists(edited_path):
                print(f"Warning: Image {edited_path} missing.")
                continue
            
            # Load Source (pre_image / GT) first for target size
            pre_img = Image.open(sample['pre_image']).convert("RGB")
            edited_img = Image.open(edited_path).convert("RGB")
            
            # Resize generated image to match GT size (e.g., 512 -> 256)
            if edited_img.size != pre_img.size:
                edited_img = edited_img.resize(pre_img.size, Image.LANCZOS)
            
            loaded_samples.append({
                'sample': sample,
                'edited_img': edited_img,
                'pre_img': pre_img,
                'instruction': sample['instruction']
            })
        except Exception as e:
            print(f"Error loading sample {sample.get('id')}: {e}")
            continue
    
    if not loaded_samples:
        print("No samples loaded. Skipping this folder.")
        return None
    
    print(f"Computing VIEScore for {len(loaded_samples)} samples with {batch_size} parallel workers...")
    
    results = []
    results_lock = threading.Lock()
    
    def process_single_sample(item):
        """Process a single sample with retry logic for transient errors."""
        last_error = None
        
        for attempt in range(MAX_RETRIES):
            try:
                viescore_result = evaluator.compute_viescore(
                    item['pre_img'], item['edited_img'], item['instruction']
                )
                
                # Check if we got valid scores (not all zeros from error)
                if (viescore_result['sc'] == 0.0 and 
                    viescore_result['pq'] == 0.0 and 
                    viescore_result['overall'] == 0.0):
                    raise ValueError("Got zero scores, likely API error")
                
                item['sample']['viescore_metrics'] = {
                    'sc': viescore_result['sc'],
                    'pq': viescore_result['pq'],
                    'overall': viescore_result['overall']
                }
                return item['sample']
                
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    # Exponential backoff with jitter
                    delay = min(BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), MAX_DELAY)
                    time.sleep(delay)
                continue
        
        print(f"Error evaluating sample {item['sample'].get('id')} after {MAX_RETRIES} retries: {last_error}")
        return None
    
    # Process samples in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        # Submit all tasks
        future_to_item = {
            executor.submit(process_single_sample, item): item 
            for item in loaded_samples
        }
        
        # Collect results with progress bar
        for future in tqdm(
            concurrent.futures.as_completed(future_to_item), 
            total=len(loaded_samples),
            desc="Computing VIEScore"
        ):
            result = future.result()
            if result is not None:
                with results_lock:
                    results.append(result)
    
    if not results:
        print("No results generated.")
        return None
    
    # Compute average scores
    avg_sc = sum(r['viescore_metrics']['sc'] for r in results) / len(results)
    avg_pq = sum(r['viescore_metrics']['pq'] for r in results) / len(results)
    avg_overall = sum(r['viescore_metrics']['overall'] for r in results) / len(results)
    
    print("\n" + "="*80)
    print("VIEScore Results:")
    print(f"  SC (Semantic Consistency): {avg_sc:.4f}")
    print(f"  PQ (Perceptual Quality):   {avg_pq:.4f}")
    print(f"  Overall:                   {avg_overall:.4f}")
    print("="*80)
    
    # Save detailed metrics
    output_data = {
        'average': {
            'viescore_sc': avg_sc,
            'viescore_pq': avg_pq,
            'viescore_overall': avg_overall
        },
        'details': results
    }
    
    out_file = os.path.join(result_dir, "viescore_metrics.json")
    with open(out_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Metrics saved to {out_file}\n")
    
    return output_data


def main():
    args = parse_args()
    
    print("VIEScore Batch Evaluation")
    print("="*80)
    print(f"Base directories: {args.base_dirs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Model: {args.viescore_model}")
    print(f"Skip existing: {args.skip_existing}")
    print("="*80)
    
    # Find all result folders
    result_folders = find_result_folders(args.base_dirs)
    
    if not result_folders:
        print("No result folders found.")
        return
    
    print(f"\nFound {len(result_folders)} result folders to evaluate.")
    
    # Initialize VIEScore evaluator
    print("\nInitializing VIEScore evaluator...")
    try:
        evaluator = VIEScoreEvaluator(model_name=args.viescore_model)
    except Exception as e:
        print(f"Error initializing VIEScore evaluator: {e}")
        print("Make sure OPENROUTER_API_KEY is set in environment or .env file")
        return
    
    # Evaluate each folder
    summary = []
    for result_dir in result_folders:
        # Skip if already evaluated
        if args.skip_existing:
            viescore_file = os.path.join(result_dir, "viescore_metrics.json")
            if os.path.exists(viescore_file):
                print(f"\nSkipping {result_dir} (already evaluated)")
                continue
        
        try:
            result = evaluate_folder(result_dir, evaluator, args.batch_size)
            if result:
                summary.append({
                    'folder': os.path.basename(result_dir),
                    'path': result_dir,
                    'average': result['average']
                })
        except Exception as e:
            print(f"Error evaluating {result_dir}: {e}")
            continue
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    for item in summary:
        print(f"\n{item['folder']}:")
        print(f"  SC:      {item['average']['viescore_sc']:.4f}")
        print(f"  PQ:      {item['average']['viescore_pq']:.4f}")
        print(f"  Overall: {item['average']['viescore_overall']:.4f}")
    
    print("\n" + "="*80)
    print("All evaluations completed.")
    print("="*80)


if __name__ == "__main__":
    main()
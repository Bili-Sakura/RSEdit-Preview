#!/usr/bin/env python3
"""
Complete missing VIEScore entries - evaluate samples that are in metadata.json
but missing from viescore_metrics.json.
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
BASE_DELAY = 1.0
MAX_DELAY = 60.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Complete missing VIEScore entries"
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="Result directory containing metadata.json and viescore_metrics.json"
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
        help="Model name for VIEScore"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only show what would be evaluated, don't actually run"
    )
    return parser.parse_args()


def find_missing_samples(result_dir):
    """Find samples in metadata.json that are missing from viescore_metrics.json."""
    metadata_path = os.path.join(result_dir, "metadata.json")
    viescore_path = os.path.join(result_dir, "viescore_metrics.json")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.json not found in {result_dir}")
    
    with open(metadata_path, 'r') as f:
        all_samples = json.load(f)
    
    # Load existing viescore data
    if os.path.exists(viescore_path):
        with open(viescore_path, 'r') as f:
            viescore_data = json.load(f)
        existing_ids = {d['id'] for d in viescore_data.get('details', [])}
    else:
        viescore_data = {'average': {}, 'details': []}
        existing_ids = set()
    
    # Find missing samples
    missing_samples = [s for s in all_samples if s['id'] not in existing_ids]
    
    return all_samples, viescore_data, missing_samples


def evaluate_missing(result_dir, missing_samples, evaluator, batch_size):
    """Evaluate missing samples with VIEScore."""
    images_dir = os.path.join(result_dir, "images")
    
    # Load images for missing samples
    print("Loading images for missing samples...")
    loaded_samples = []
    for sample in tqdm(missing_samples, desc="Loading"):
        try:
            edited_path = os.path.join(images_dir, sample['filename'])
            if not os.path.exists(edited_path):
                print(f"Warning: Image {edited_path} missing. Skipping.")
                continue
            
            pre_img = Image.open(sample['pre_image']).convert("RGB")
            edited_img = Image.open(edited_path).convert("RGB")
            
            # Resize generated image to match GT size if needed
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
        print("No samples could be loaded.")
        return []
    
    print(f"Evaluating {len(loaded_samples)} missing samples with {batch_size} parallel workers...")
    
    results = []
    results_lock = threading.Lock()
    
    def process_single_sample(item):
        """Process a single sample with retry logic."""
        last_error = None
        
        for attempt in range(MAX_RETRIES):
            try:
                viescore_result = evaluator.compute_viescore(
                    item['pre_img'], item['edited_img'], item['instruction']
                )
                
                # Check if we got valid scores
                if (viescore_result['sc'] == 0.0 and 
                    viescore_result['pq'] == 0.0 and 
                    viescore_result['overall'] == 0.0):
                    raise ValueError("Got zero scores, likely API error")
                
                result = item['sample'].copy()
                result['status'] = 'success'
                result['viescore_metrics'] = {
                    'sc': viescore_result['sc'],
                    'pq': viescore_result['pq'],
                    'overall': viescore_result['overall']
                }
                return result
                
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    delay = min(BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), MAX_DELAY)
                    time.sleep(delay)
                continue
        
        print(f"Failed sample {item['sample'].get('id')} after {MAX_RETRIES} retries: {last_error}")
        return None
    
    # Process samples in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        future_to_item = {
            executor.submit(process_single_sample, item): item 
            for item in loaded_samples
        }
        
        for future in tqdm(
            concurrent.futures.as_completed(future_to_item), 
            total=len(loaded_samples),
            desc="Computing VIEScore"
        ):
            result = future.result()
            if result is not None:
                with results_lock:
                    results.append(result)
    
    return results


def main():
    args = parse_args()
    
    print("VIEScore Completion Script")
    print("="*80)
    print(f"Result directory: {args.result_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Model: {args.viescore_model}")
    print(f"Dry run: {args.dry_run}")
    print("="*80)
    
    # Find missing samples
    all_samples, viescore_data, missing_samples = find_missing_samples(args.result_dir)
    
    print(f"\nTotal samples in metadata.json: {len(all_samples)}")
    print(f"Existing in viescore_metrics.json: {len(viescore_data.get('details', []))}")
    print(f"Missing samples to evaluate: {len(missing_samples)}")
    
    if not missing_samples:
        print("\nNo missing samples. VIEScore metrics are complete!")
        return
    
    if args.dry_run:
        print("\nDRY RUN - would evaluate these samples:")
        for sample in missing_samples[:10]:
            print(f"  - ID {sample['id']}: {sample['filename']}")
        if len(missing_samples) > 10:
            print(f"  ... and {len(missing_samples) - 10} more")
        return
    
    # Initialize VIEScore evaluator
    print("\nInitializing VIEScore evaluator...")
    try:
        evaluator = VIEScoreEvaluator(model_name=args.viescore_model)
    except Exception as e:
        print(f"Error initializing VIEScore evaluator: {e}")
        print("Make sure OPENROUTER_API_KEY is set in environment or .env file")
        return
    
    # Evaluate missing samples
    new_results = evaluate_missing(args.result_dir, missing_samples, evaluator, args.batch_size)
    
    if not new_results:
        print("No new results obtained.")
        return
    
    print(f"\nSuccessfully evaluated {len(new_results)}/{len(missing_samples)} missing samples.")
    
    # Merge results
    existing_details = viescore_data.get('details', [])
    all_details = existing_details + new_results
    
    # Sort by id for consistency
    all_details.sort(key=lambda x: x['id'])
    
    # Recompute averages
    success_entries = [d for d in all_details if d.get('status') == 'success' and 'viescore_metrics' in d]
    if success_entries:
        avg_sc = sum(d['viescore_metrics']['sc'] for d in success_entries) / len(success_entries)
        avg_pq = sum(d['viescore_metrics']['pq'] for d in success_entries) / len(success_entries)
        avg_overall = sum(d['viescore_metrics']['overall'] for d in success_entries) / len(success_entries)
        
        viescore_data['average'] = {
            'viescore_sc': avg_sc,
            'viescore_pq': avg_pq,
            'viescore_overall': avg_overall
        }
    
    viescore_data['details'] = all_details
    
    # Save updated data
    viescore_path = os.path.join(args.result_dir, "viescore_metrics.json")
    with open(viescore_path, 'w') as f:
        json.dump(viescore_data, f, indent=2)
    
    print("\n" + "="*80)
    print("RESULTS:")
    print(f"  Previous entries: {len(existing_details)}")
    print(f"  New entries: {len(new_results)}")
    print(f"  Total entries: {len(all_details)}")
    print(f"\n  Updated averages:")
    print(f"    SC:      {viescore_data['average']['viescore_sc']:.4f}")
    print(f"    PQ:      {viescore_data['average']['viescore_pq']:.4f}")
    print(f"    Overall: {viescore_data['average']['viescore_overall']:.4f}")
    print(f"\nSaved to: {viescore_path}")
    print("="*80)


if __name__ == "__main__":
    main()

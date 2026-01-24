#!/usr/bin/env python3
"""
Fix skipped VIEScore entries - re-evaluate and update with correct results.
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
        description="Fix skipped VIEScore entries by re-evaluating them"
    )
    parser.add_argument(
        "--base_dirs",
        type=str,
        nargs="+",
        default=[
            # "/path/to/results"
            "/path/to/results"
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
        help="Model name for VIEScore"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only show what would be fixed, don't actually run"
    )
    return parser.parse_args()


def find_files_with_skipped(base_dirs):
    """Find all viescore_metrics.json files that have skipped entries."""
    files_to_fix = []
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"Warning: Base directory {base_dir} does not exist. Skipping.")
            continue
        
        print(f"Scanning {base_dir}...")
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                viescore_path = os.path.join(item_path, "viescore_metrics.json")
                if os.path.exists(viescore_path):
                    with open(viescore_path, 'r') as f:
                        data = json.load(f)
                    
                    skipped_count = sum(1 for d in data.get('details', []) 
                                       if d.get('status') != 'success')
                    if skipped_count > 0:
                        files_to_fix.append({
                            'path': viescore_path,
                            'folder': item_path,
                            'skipped_count': skipped_count
                        })
                        print(f"  Found: {item} ({skipped_count} skipped)")
    
    return files_to_fix


def fix_file(viescore_path, folder_path, evaluator, batch_size, dry_run=False):
    """Fix skipped entries in a single viescore_metrics.json file."""
    print(f"\n{'='*80}")
    print(f"Fixing: {viescore_path}")
    print(f"{'='*80}")
    
    # Load existing data
    with open(viescore_path, 'r') as f:
        data = json.load(f)
    
    details = data.get('details', [])
    
    # Find skipped entries
    skipped_entries = [(i, d) for i, d in enumerate(details) if d.get('status') != 'success']
    
    if not skipped_entries:
        print("No skipped entries found.")
        return 0
    
    print(f"Found {len(skipped_entries)} skipped entries to fix.")
    
    if dry_run:
        print("DRY RUN - would fix these entries:")
        for idx, entry in skipped_entries[:5]:
            print(f"  - ID {entry.get('id')}: {entry.get('filename')}")
        if len(skipped_entries) > 5:
            print(f"  ... and {len(skipped_entries) - 5} more")
        return len(skipped_entries)
    
    images_dir = os.path.join(folder_path, "images")
    
    # Load images for skipped entries
    print("Loading images for skipped entries...")
    loaded_samples = []
    for idx, entry in tqdm(skipped_entries, desc="Loading"):
        try:
            edited_path = os.path.join(images_dir, entry['filename'])
            if not os.path.exists(edited_path):
                print(f"Warning: Image {edited_path} missing. Cannot fix.")
                continue
            
            pre_img = Image.open(entry['pre_image']).convert("RGB")
            edited_img = Image.open(edited_path).convert("RGB")
            
            # Resize generated image to match GT size (e.g., 512 -> 256)
            if edited_img.size != pre_img.size:
                edited_img = edited_img.resize(pre_img.size, Image.LANCZOS)
            
            loaded_samples.append({
                'idx': idx,
                'entry': entry,
                'edited_img': edited_img,
                'pre_img': pre_img,
                'instruction': entry['instruction']
            })
        except Exception as e:
            print(f"Error loading sample {entry.get('id')}: {e}")
            continue
    
    if not loaded_samples:
        print("No samples could be loaded. Skipping this file.")
        return 0
    
    print(f"Re-evaluating {len(loaded_samples)} samples with {batch_size} parallel workers...")
    
    fixed_count = 0
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
                
                return {
                    'idx': item['idx'],
                    'viescore_metrics': {
                        'sc': viescore_result['sc'],
                        'pq': viescore_result['pq'],
                        'overall': viescore_result['overall']
                    }
                }
                
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    delay = min(BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), MAX_DELAY)
                    time.sleep(delay)
                continue
        
        print(f"Failed to fix sample {item['entry'].get('id')} after {MAX_RETRIES} retries: {last_error}")
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
            desc="Re-evaluating"
        ):
            result = future.result()
            if result is not None:
                with results_lock:
                    # Update the entry in details
                    idx = result['idx']
                    details[idx]['status'] = 'success'
                    details[idx]['viescore_metrics'] = result['viescore_metrics']
                    fixed_count += 1
    
    if fixed_count == 0:
        print("No entries were fixed.")
        return 0
    
    # Recompute averages (only from success entries)
    success_entries = [d for d in details if d.get('status') == 'success' and 'viescore_metrics' in d]
    if success_entries:
        avg_sc = sum(d['viescore_metrics']['sc'] for d in success_entries) / len(success_entries)
        avg_pq = sum(d['viescore_metrics']['pq'] for d in success_entries) / len(success_entries)
        avg_overall = sum(d['viescore_metrics']['overall'] for d in success_entries) / len(success_entries)
        
        data['average'] = {
            'viescore_sc': avg_sc,
            'viescore_pq': avg_pq,
            'viescore_overall': avg_overall
        }
    
    # Save updated data
    with open(viescore_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Fixed {fixed_count}/{len(skipped_entries)} entries. Saved to {viescore_path}")
    return fixed_count


def main():
    args = parse_args()
    
    print("VIEScore Skipped Entry Fixer")
    print("="*80)
    print(f"Base directories: {args.base_dirs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Model: {args.viescore_model}")
    print(f"Dry run: {args.dry_run}")
    print("="*80)
    
    # Find files with skipped entries
    files_to_fix = find_files_with_skipped(args.base_dirs)
    
    if not files_to_fix:
        print("\nNo files with skipped entries found.")
        return
    
    total_skipped = sum(f['skipped_count'] for f in files_to_fix)
    print(f"\nFound {len(files_to_fix)} files with {total_skipped} total skipped entries.")
    
    if args.dry_run:
        print("\nDRY RUN - no changes will be made.")
        return
    
    # Initialize VIEScore evaluator
    print("\nInitializing VIEScore evaluator...")
    try:
        evaluator = VIEScoreEvaluator(model_name=args.viescore_model)
    except Exception as e:
        print(f"Error initializing VIEScore evaluator: {e}")
        print("Make sure OPENROUTER_API_KEY is set in environment or .env file")
        return
    
    # Fix each file
    total_fixed = 0
    for file_info in files_to_fix:
        try:
            fixed = fix_file(
                file_info['path'], 
                file_info['folder'], 
                evaluator, 
                args.batch_size
            )
            total_fixed += fixed
        except Exception as e:
            print(f"Error fixing {file_info['path']}: {e}")
            continue
    
    print("\n" + "="*80)
    print(f"DONE - Fixed {total_fixed}/{total_skipped} skipped entries.")
    print("="*80)


if __name__ == "__main__":
    main()

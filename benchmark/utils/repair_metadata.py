#!/usr/bin/env python3
"""
Rebuild metadata.json for a results folder from dataset annotations.
"""

import argparse
import json
import os
from typing import List, Dict, Any


def generate_filename(post_image_path: str, model_name: str) -> str:
    base_name = os.path.splitext(os.path.basename(post_image_path))[0]
    sanitized_model_name = model_name.replace(" ", "_").replace(".", "_")
    return f"{base_name}_{sanitized_model_name}.png"


def load_levircc_dataset(json_path: str, base_dir: str) -> List[Dict[str, Any]]:
    with open(json_path, "r") as f:
        data = json.load(f)

    samples = []
    for img_info in data["images"]:
        if img_info.get("split") != "test":
            continue
        filename = img_info["filename"]
        filepath = img_info["filepath"]
        pre_image = os.path.join(base_dir, filepath, "A", filename)
        post_image = os.path.join(base_dir, filepath, "B", filename)
        if img_info.get("sentences"):
            instruction = img_info["sentences"][0]["raw"].strip()
        else:
            instruction = "Apply the change described in the image pair."
        samples.append(
            {
                "pre_image": pre_image,
                "post_image": post_image,
                "instruction": instruction,
            }
        )
    return samples


def load_second_cc_aug_dataset(json_path: str, base_dir: str) -> List[Dict[str, Any]]:
    with open(json_path, "r") as f:
        data = json.load(f)

    samples = []
    for img_info in data["images"]:
        if img_info.get("split") != "test":
            continue
        filename = img_info["filename"]
        filepath = img_info["filepath"]
        pre_image = os.path.join(base_dir, filepath, "rgb", "A", filename)
        post_image = os.path.join(base_dir, filepath, "rgb", "B", filename)
        if img_info.get("sentences"):
            instruction = img_info["sentences"][0]["raw"].strip()
        else:
            instruction = "Apply the change described in the image pair."
        samples.append(
            {
                "pre_image": pre_image,
                "post_image": post_image,
                "instruction": instruction,
            }
        )
    return samples


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild metadata.json for a results folder"
    )
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--dataset_type", type=str, required=True, choices=["levircc", "second_cc_aug"])
    parser.add_argument("--dataset_json", type=str, required=True)
    parser.add_argument("--dataset_base_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    images_dir = os.path.join(args.results_dir, "images")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"images/ not found in {args.results_dir}")

    if args.dataset_type == "levircc":
        samples = load_levircc_dataset(args.dataset_json, args.dataset_base_dir)
    else:
        samples = load_second_cc_aug_dataset(args.dataset_json, args.dataset_base_dir)

    metadata = []
    success_count = 0
    missing_count = 0

    for idx, sample in enumerate(samples):
        filename = generate_filename(sample["post_image"], args.model_name)
        edited_path = os.path.join(images_dir, filename)
        status = "success" if os.path.exists(edited_path) else "missing"
        if status == "success":
            success_count += 1
        else:
            missing_count += 1
        metadata.append(
            {
                "id": idx,
                "filename": filename,
                "pre_image": sample["pre_image"],
                "post_image": sample["post_image"],
                "instruction": sample["instruction"],
                "status": status,
            }
        )

    print(f"Samples: {len(metadata)}")
    print(f"  success: {success_count}")
    print(f"  missing: {missing_count}")

    if args.dry_run:
        print("Dry run - not writing metadata.json")
        return

    out_path = os.path.join(args.results_dir, "metadata.json")
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

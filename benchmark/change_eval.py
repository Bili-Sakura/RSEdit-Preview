import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import albumentations.pytorch
import ever as er
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from torchange.metrics.xview2 import (
    _accumulate_dam,
    _accumulate_loc,
    mixed_score,
    parse_prediction_v1,
)
from torchange.models.changen2 import changestar_1x256


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _setup_logger() -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )
    return logging.getLogger("change_eval")


def _load_checkpoint(model: torch.nn.Module, ckpt_path: str, logger: logging.Logger):
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state and isinstance(state["model"], dict):
            state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("Missing keys when loading checkpoint: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys when loading checkpoint: %s", unexpected)
    return model


class RSCCChangeDataset(Dataset):
    """Dataset for Remote Sensing Change Detection evaluation.
    
    Data Structure (RSCC-RSEdit-Test-Split format):
    -----------------------------------------------
    Expects a JSONL file where each line contains:
        - pre_image: path to pre-disaster image (512x512 RGB PNG)
        - post_image: path to post-disaster image (512x512 RGB PNG)
        - mask: path to damage mask (512x512 grayscale PNG, values 0-4)
        - change_caption: text description of changes (optional)
        - filename: output filename identifier
    
    Damage Mask Values:
        - 0: background
        - 1: no-damage
        - 2: minor-damage
        - 3: major-damage
        - 4: destroyed
    
    Model Output (ChangeStar s1c5):
        - t1_semantic_prediction: (B, 1, H, W) - localization probability (sigmoid)
        - change_prediction: (B, 5, H, W) - damage class probabilities (softmax)
    """
    
    def __init__(self, data_path: str, max_items: int = None):
        """
        Args:
            data_path: Path to JSONL file with pre_image, post_image, mask paths
            max_items: Maximum number of items to load (for debugging)
        """
        self.samples: List[Dict] = []
        
        # Support both JSON and JSONL formats
        with open(data_path, "r") as f:
            first_char = f.read(1)
            f.seek(0)
            
            if first_char == "[":
                # JSON array format (legacy)
                data = json.load(f)
                for obj in data:
                    self._add_sample(obj)
                    if max_items and len(self.samples) >= max_items:
                        break
            else:
                # JSONL format (new)
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        self._add_sample(obj)
                        if max_items and len(self.samples) >= max_items:
                            break

        self.transforms = A.Compose(
            [A.Normalize(), A.pytorch.ToTensorV2()],
            additional_targets={"image2": "image"},
        )
    
    def _add_sample(self, obj: Dict):
        """Add a sample from parsed JSON object."""
        pre = obj.get("pre_image")
        post = obj.get("post_image")
        if not (pre and post):
            return
        # Skip failed items (legacy format compatibility)
        status = obj.get("status")
        if status and status != "success":
            return
        
        entry = {
            "pre_image": pre,
            "post_image": post,
            "mask": obj.get("mask"),  # Direct mask path (new format)
            "filename": obj.get("filename") or Path(post).name,
            "change_caption": obj.get("change_caption", ""),
        }
        self.samples.append(entry)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pre_path = Path(sample["pre_image"])
        post_path = Path(sample["post_image"])
        pre_img = np.array(Image.open(pre_path).convert("RGB"))
        post_img = np.array(Image.open(post_path).convert("RGB"))

        # Load mask directly from path in JSONL
        mask_path = Path(sample["mask"])
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        damage_mask = np.array(Image.open(mask_path)).astype(np.uint8)

        loc_mask = (damage_mask > 0).astype(np.uint8)
        data = self.transforms(image=pre_img, image2=post_img)
        image_tensor = torch.cat([data["image"], data["image2"]], dim=0)

        return (
            image_tensor,
            (
                torch.from_numpy(loc_mask),
                torch.from_numpy(damage_mask),
            ),
            sample["filename"],
            sample.get("change_caption", ""),
        )


class GeneratedResultsDataset(Dataset):
    """Dataset for evaluating generated post-disaster images against ground truth.
    
    This dataset loads:
        - Pre-disaster images from original paths
        - Generated post-disaster images from a results folder
        - Ground truth masks from RSCC-RSEdit-Test-Split
    
    Expected structure of results_dir:
        results_dir/
        ├── metadata.json  (with pre_image, filename, status fields)
        └── images/
            └── {filename}  (generated post-disaster images)
    
    Ground truth structure:
        gt_dir/
        └── masks/
            └── {original_post_disaster_name}.png
    """
    
    def __init__(
        self, 
        results_dir: str, 
        gt_dir: str = "/path/to/dataset",
        max_items: int = None
    ):
        """
        Args:
            results_dir: Path to generated results folder (e.g., text2earth, magic-brush)
            gt_dir: Path to ground truth folder with masks/ subdirectory
            max_items: Maximum number of items to load (for debugging)
        """
        self.samples: List[Dict] = []
        self.results_dir = Path(results_dir)
        self.gt_masks_dir = Path(gt_dir) / "masks"
        
        # Load metadata
        metadata_path = self.results_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {results_dir}")
        
        with open(metadata_path, "r") as f:
            data = json.load(f)
        
        for obj in data:
            status = obj.get("status")
            # Include both "success" and "skipped" entries (skipped images still exist and should be evaluated)
            if status and status not in ("success", "skipped"):
                continue
            
            pre_image = obj.get("pre_image")
            filename = obj.get("filename")
            
            if not (pre_image and filename):
                continue
            
            # Generated image path
            gen_image = self.results_dir / "images" / filename
            
            # Derive ground truth mask name from pre_image
            # pre_image: .../{event}_{id}_pre_disaster_{part}.png
            # mask: {event}_{id}_post_disaster_{part}.png
            pre_name = Path(pre_image).name
            mask_name = pre_name.replace("_pre_disaster", "_post_disaster")
            mask_path = self.gt_masks_dir / mask_name
            
            entry = {
                "pre_image": pre_image,
                "post_image": str(gen_image),  # Generated image
                "mask": str(mask_path),
                "filename": filename,
                "instruction": obj.get("instruction", ""),
            }
            self.samples.append(entry)
            
            if max_items and len(self.samples) >= max_items:
                break
        
        self.transforms = A.Compose(
            [A.Normalize(), A.pytorch.ToTensorV2()],
            additional_targets={"image2": "image"},
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pre_path = Path(sample["pre_image"])
        post_path = Path(sample["post_image"])
        mask_path = Path(sample["mask"])
        
        # Load images
        pre_img = np.array(Image.open(pre_path).convert("RGB"))
        post_img = np.array(Image.open(post_path).convert("RGB"))
        
        # Load ground truth mask
        if not mask_path.exists():
            raise FileNotFoundError(f"GT mask not found: {mask_path}")
        damage_mask = np.array(Image.open(mask_path)).astype(np.uint8)

        loc_mask = (damage_mask > 0).astype(np.uint8)
        data = self.transforms(image=pre_img, image2=post_img)
        image_tensor = torch.cat([data["image"], data["image2"]], dim=0)

        return (
            image_tensor,
            (
                torch.from_numpy(loc_mask),
                torch.from_numpy(damage_mask),
            ),
            sample["filename"],
            sample.get("instruction", ""),
        )


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate Changen2 ChangeStar1x256 on RSCC pre/post pairs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate ground truth images:
  python change_eval.py --data_path /path/to/dataset

  # Evaluate generated results (e.g., text2earth):
  python change_eval.py --results_dir /path/to/results

  # Evaluate with custom ground truth location:
  python change_eval.py --results_dir /path/to/results --gt_dir /path/to/dataset
        """
    )
    
    # Data source (mutually exclusive)
    data_group = p.add_mutually_exclusive_group(required=False)
    data_group.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to JSONL/JSON file with pre_image, post_image, mask paths (for ground truth eval)",
    )
    data_group.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Path to generated results folder with metadata.json and images/ (for generated eval)",
    )
    
    p.add_argument(
        "--gt_dir",
        type=str,
        default="/path/to/dataset",
        help="Path to ground truth folder with masks/ subdirectory (used with --results_dir)",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="/path/to/model",
    )
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results. Defaults to {results_dir} or 'outputs/change_eval'",
    )
    p.add_argument(
        "--no_save", 
        action="store_true",
        help="Do not save predicted change masks",
    )
    p.add_argument(
        "--save_masks_dir",
        type=str,
        default=None,
        help="Custom directory to save predicted change masks (overrides default change_masks/)",
    )
    p.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU device to use (e.g., '0', '1', 'cuda:0', or 'cpu'). If not specified, uses auto_device().",
    )
    p.add_argument(
        "--use_gt_loc",
        action="store_true",
        help="Use ground truth building location mask instead of model-predicted localization. "
             "This evaluates damage detection quality independently of building localization accuracy.",
    )
    return p.parse_args()


@torch.no_grad()
def evaluate(
    model,
    dataloader: DataLoader,
    logger: logging.Logger,
    output_dir: Path,
    device: torch.device,
    save_preds: bool = False,
    save_masks_dir: str = None,
    use_gt_loc: bool = False,
):
    loc_metric_op = er.metric.PixelMetric(2, str(output_dir), logger=logger)
    damage_metric_op = er.metric.PixelMetric(5, str(output_dir), logger=logger)

    # Determine where to save predicted masks
    if save_preds:
        if save_masks_dir:
            preds_dir = Path(save_masks_dir)
        else:
            preds_dir = output_dir / "pred_masks"
        preds_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving predicted change masks to: %s", preds_dir)
    else:
        preds_dir = None

    for imgs, masks, filenames, _captions in tqdm(dataloader):
        imgs = imgs.to(device)
        gt_loc, gt_dam = masks
        gt_loc = gt_loc.to(device)
        gt_dam = gt_dam.to(device)

        pred = model(imgs)
        loc_pred, dam_pred = parse_prediction_v1(pred)
        
        # Fix broadcasting issue:
        # - loc_pred from parse_prediction_v1 has shape (B, 1, H, W) from t1_semantic_prediction > 0.5
        # - dam_pred has shape (B, H, W) from change_prediction.argmax(dim=1)
        # Without squeeze, (B, 1, H, W) * (B, H, W) broadcasts incorrectly to (B, B, H, W)
        # We need to squeeze loc_pred to (B, H, W) for proper element-wise multiplication
        loc_pred = loc_pred.squeeze(1)  # (B, 1, H, W) -> (B, H, W)
        
        if use_gt_loc:
            # Use ground truth building location mask instead of model prediction
            # This evaluates damage detection independently of building localization
            dam_pred = gt_loc * dam_pred  # (B, H, W) * (B, H, W) -> (B, H, W)
        else:
            dam_pred = loc_pred * dam_pred  # (B, H, W) * (B, H, W) -> (B, H, W)

        for b in range(imgs.shape[0]):
            if not use_gt_loc:
                # Only accumulate loc metrics when not using ground truth location
                _accumulate_loc(loc_metric_op, gt_loc[b].cpu(), loc_pred[b].cpu())
            _accumulate_dam(damage_metric_op, gt_dam[b].cpu(), dam_pred[b].cpu())

            if preds_dir:
                # Save predicted change mask (dam_pred has shape (H, W) after squeeze)
                out_mask = dam_pred[b].cpu().numpy().astype(np.uint8)
                Image.fromarray(out_mask).save(preds_dir / f"{filenames[b]}")

    er.dist.synchronize()
    dam_tb = damage_metric_op.summary_all()
    
    if use_gt_loc:
        # When using ground truth location, we only compute damage F1
        # No loc_f1 or mixed score needed - directly compute from damage table
        nodamage_f1 = dam_tb.f1(1)
        minor_f1 = dam_tb.f1(2)
        major_f1 = dam_tb.f1(3)
        destroyed_f1 = dam_tb.f1(4)
        
        # Harmonic mean of per-class F1 scores (same as xView2 scoring)
        harmonic_mean = lambda xs: len(xs) / sum((x + 1e-6) ** -1 for x in xs)
        dam_f1 = harmonic_mean([nodamage_f1, minor_f1, major_f1, destroyed_f1])
        
        metrics = {
            "dam_f1_gt_loc": dam_f1,
            "per_class_f1_gt_loc": {
                "no_damage": nodamage_f1,
                "minor": minor_f1,
                "major": major_f1,
                "destroyed": destroyed_f1,
            },
        }
        output_file = output_dir / "metrics_gt_loc.json"
    else:
        loc_tb = loc_metric_op.summary_all()
        loc_f1, dam_f1, final_f1, dam_f1s = mixed_score(loc_tb, dam_tb)

        metrics = {
            "loc_f1": loc_f1,
            "dam_f1": dam_f1,
            "final_f1": final_f1,
            "per_class_f1": {
                "no_damage": dam_f1s[0],
                "minor": dam_f1s[1],
                "major": dam_f1s[2],
                "destroyed": dam_f1s[3],
            },
        }
        output_file = output_dir / "metrics.json"
    
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    return metrics


def main():
    args = parse_args()
    logger = _setup_logger()

    # Determine dataset mode and create appropriate dataset
    if args.results_dir:
        # Evaluate generated results
        logger.info("Evaluating generated results from: %s", args.results_dir)
        logger.info("Using ground truth masks from: %s", args.gt_dir)
        dataset = GeneratedResultsDataset(
            results_dir=args.results_dir,
            gt_dir=args.gt_dir,
            max_items=args.max_items,
        )
        # Default output_dir to results_dir if not specified
        output_dir = Path(args.output_dir) if args.output_dir else Path(args.results_dir)
    elif args.data_path:
        # Evaluate ground truth
        logger.info("Evaluating ground truth from: %s", args.data_path)
        dataset = RSCCChangeDataset(args.data_path, max_items=args.max_items)
        output_dir = Path(args.output_dir) if args.output_dir else Path("outputs/change_eval")
    else:
        # Default to ground truth evaluation
        default_path = "/path/to/dataset",
        logger.info("No data source specified, using default: %s", default_path)
        dataset = RSCCChangeDataset(default_path, max_items=args.max_items)
        output_dir = Path(args.output_dir) if args.output_dir else Path("outputs/change_eval")
    
    output_dir.mkdir(parents=True, exist_ok=True)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
    )

    logger.info("Loaded %d samples", len(dataset))
    
    # Determine device
    if args.gpu is not None:
        if args.gpu.lower() == "cpu":
            device = torch.device("cpu")
        elif args.gpu.isdigit():
            device = torch.device(f"cuda:{args.gpu}")
        elif args.gpu.startswith("cuda:"):
            device = torch.device(args.gpu)
        else:
            raise ValueError(f"Invalid GPU specification: {args.gpu}")
        logger.info("Using device: %s", device)
    else:
        device = er.auto_device()
        logger.info("Using auto device: %s", device)
    
    # Initialize model architecture only (no pretrained weights to avoid HuggingFace download)
    # Suppress EVER's misleading log (shows None for pretrained/finetuned since we load locally)
    ever_logger = logging.getLogger("EVER")
    original_level = ever_logger.level
    ever_logger.setLevel(logging.WARNING)
    model = changestar_1x256('vitb', 's1c5', changen2_pretrained=None)
    ever_logger.setLevel(original_level)
    
    # Load checkpoint from local path
    logger.info("Loading checkpoint from local path: %s", args.checkpoint)
    model = _load_checkpoint(model, args.checkpoint, logger)
    logger.info(
        "Model loaded: architecture=changestar_1x256 | backbone=vitb | "
        "modeling_type=s1c5 | pretrained=Changen2-S0-1.2M | finetuned_on=xView2"
    )
    model.eval()
    model.to(device)

    metrics = evaluate(
        model, dataloader, logger, output_dir, device, 
        save_preds=not args.no_save,
        save_masks_dir=args.save_masks_dir,
        use_gt_loc=args.use_gt_loc,
    )
    logger.info("Metrics: %s", metrics)


if __name__ == "__main__":
    # CLI examples:
    #
    # Evaluate ground truth (original pre/post pairs):
    #   python change_eval.py --data_path /path/to/dataset
    #
    # Evaluate generated results (e.g., text2earth, magic-brush):
    #   python change_eval.py --results_dir /path/to/results
    #   python change_eval.py --results_dir /path/to/results
    #
    # Save predicted change masks:
    #   python change_eval.py --results_dir /path/to/results --save_preds
    #   python change_eval.py --results_dir /path/to/results --save_preds --save_masks_dir /path/to/masks
    #
    main()

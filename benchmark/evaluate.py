import argparse
import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer, CLIPImageProcessor, AutoModel, AutoImageProcessor
import lpips
from torch.nn import functional as F

# Import VIEScore evaluator
try:
    from viescore_openrouter import VIEScoreEvaluator
    VIESCORE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: VIEScore not available: {e}")
    VIESCORE_AVAILABLE = False

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated images against test set (supports RSCC, LevirCC, SECOND-CC-AUG)")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing generated images and metadata.json")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for metric computation")
    parser.add_argument("--use_viescore", action="store_true", help="Enable VIEScore evaluation (requires OpenRouter API key)")
    parser.add_argument("--viescore_model", type=str, default="google/gemini-2.5-flash", 
                        help="OpenRouter model name for VIEScore")
    return parser.parse_args()

class RSCCEvaluator:
    def __init__(self, device="cuda", use_viescore=False, viescore_model="google/gemini-2.5-flash"):
        self.device = device
        self.use_viescore = use_viescore
        self.viescore_model_name = viescore_model
        self.init_metrics()

    def init_metrics(self):
        print("Loading metrics models...")
        # DGTRS-CLIP
        try:
            dgtrs_path = "/path/to/model"
            self.dgtrs_model = CLIPModel.from_pretrained(dgtrs_path).to(self.device)
            self.dgtrs_tokenizer = CLIPTokenizer.from_pretrained(dgtrs_path)
            self.dgtrs_image_processor = CLIPImageProcessor.from_pretrained(dgtrs_path)
            self.dgtrs_model.eval()
            print("DGTRS-CLIP loaded.")
        except Exception as e:
            print(f"Warning: Failed to load DGTRS-CLIP: {e}")
            self.dgtrs_model = None

        # DINOv3-Sat
        try:
            dinov3_path = "/path/to/model"
            self.dinov3_model = AutoModel.from_pretrained(dinov3_path).to(self.device)
            self.dinov3_processor = AutoImageProcessor.from_pretrained(dinov3_path)
            self.dinov3_model.eval()
            print("DINOv3-Sat loaded.")
        except Exception as e:
            print(f"Warning: Failed to load DINOv3: {e}")
            self.dinov3_model = None

        # LPIPS
        try:
            self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
            print("LPIPS loaded.")
        except Exception as e:
            print(f"Warning: Failed to load LPIPS: {e}")
            self.lpips_fn = None
        
        # VIEScore (API-based)
        if self.use_viescore:
            if VIESCORE_AVAILABLE:
                try:
                    self.viescore_evaluator = VIEScoreEvaluator(model_name=self.viescore_model_name)
                    print(f"VIEScore loaded with model: {self.viescore_model_name}")
                except Exception as e:
                    print(f"Warning: Failed to load VIEScore: {e}")
                    self.viescore_evaluator = None
            else:
                print("Warning: VIEScore requested but not available (import failed)")
                self.viescore_evaluator = None
        else:
            self.viescore_evaluator = None

    @torch.no_grad()
    def compute_dgtrs_score(self, image, text):
        """Compute DGTRS-CLIP score for a single (image, text) pair."""
        scores = self.compute_dgtrs_score_batch([image], [text])
        return scores[0] if scores else 0.0
    
    @torch.no_grad()
    def compute_dgtrs_score_batch(self, images, texts):
        """Compute DGTRS-CLIP scores for a batch of (image, text) pairs."""
        if self.dgtrs_model is None:
            return [0.0] * len(images)
        
        text_inputs = self.dgtrs_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True)
        image_inputs = self.dgtrs_image_processor(images=images, return_tensors="pt")
        
        inputs = {
            "input_ids": text_inputs.input_ids.to(self.device),
            "attention_mask": text_inputs.attention_mask.to(self.device),
            "pixel_values": image_inputs.pixel_values.to(self.device)
        }
        
        outputs = self.dgtrs_model(**inputs)
        
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
        
        # Compute pairwise similarities (diagonal for matching pairs)
        similarity = torch.matmul(image_embeds, text_embeds.t())
        scores = similarity.diagonal().cpu().tolist()
        return scores

    @torch.no_grad()
    def compute_dinov3_similarity(self, image1, image2):
        """Compute DINOv3-Sat similarity for a single (image1, image2) pair."""
        scores = self.compute_dinov3_similarity_batch([image1], [image2])
        return scores[0] if scores else 0.0
    
    @torch.no_grad()
    def compute_dinov3_similarity_batch(self, images1, images2):
        """Compute DINOv3-Sat similarities for a batch of (image1, image2) pairs."""
        if self.dinov3_model is None:
            return [0.0] * len(images1)
        
        # Process both image sets
        inputs1 = self.dinov3_processor(images=images1, return_tensors="pt")
        inputs2 = self.dinov3_processor(images=images2, return_tensors="pt")
        
        inputs1 = {k: v.to(self.device) for k, v in inputs1.items()}
        inputs2 = {k: v.to(self.device) for k, v in inputs2.items()}
        
        outputs1 = self.dinov3_model(**inputs1)
        outputs2 = self.dinov3_model(**inputs2)
        
        feat1 = outputs1.last_hidden_state[:, 0, :]
        feat2 = outputs2.last_hidden_state[:, 0, :]
        
        feat1 = feat1 / feat1.norm(p=2, dim=-1, keepdim=True)
        feat2 = feat2 / feat2.norm(p=2, dim=-1, keepdim=True)
        
        # Compute pairwise similarities (diagonal for matching pairs)
        similarity = torch.matmul(feat1, feat2.t())
        scores = similarity.diagonal().cpu().tolist()
        return scores

    @torch.no_grad()
    def compute_lpips(self, image1, image2):
        """Compute LPIPS distance for a single (image1, image2) pair."""
        scores = self.compute_lpips_batch([image1], [image2])
        return scores[0] if scores else 0.0
    
    @torch.no_grad()
    def compute_lpips_batch(self, images1, images2):
        """Compute LPIPS distances for a batch of (image1, image2) pairs."""
        if self.lpips_fn is None:
            return [0.0] * len(images1)
        
        def preprocess(pil_img):
            img = np.array(pil_img).astype(np.float32) / 255.0
            img = img * 2.0 - 1.0
            img = torch.from_numpy(img).permute(2, 0, 1)
            return img.to(self.device)
        
        # Preprocess all images
        tensors1 = torch.stack([preprocess(img) for img in images1])
        tensors2 = torch.stack([preprocess(img) for img in images2])
        
        # Compute LPIPS for batch
        dists = self.lpips_fn(tensors1, tensors2)
        scores = dists.squeeze().cpu().tolist()
        # Handle single element case
        if isinstance(scores, float):
            scores = [scores]
        return scores
    
    def compute_viescore(self, pre_image, edited_image, instruction):
        """Compute VIEScore for a single (pre_image, edited_image, instruction) triple."""
        if self.viescore_evaluator is None:
            return {'sc': 0.0, 'pq': 0.0, 'overall': 0.0}
        
        try:
            scores = self.viescore_evaluator.compute_viescore(pre_image, edited_image, instruction)
            return scores
        except Exception as e:
            print(f"Error computing VIEScore: {e}")
            return {'sc': 0.0, 'pq': 0.0, 'overall': 0.0}
    
    def compute_viescore_batch(self, pre_images, edited_images, instructions):
        """Compute VIEScore for a batch of samples (API processes sequentially)."""
        if self.viescore_evaluator is None:
            return [{'sc': 0.0, 'pq': 0.0, 'overall': 0.0}] * len(pre_images)
        
        try:
            scores_list = self.viescore_evaluator.compute_viescore_batch(
                pre_images, edited_images, instructions
            )
            return scores_list
        except Exception as e:
            print(f"Error computing VIEScore batch: {e}")
            return [{'sc': 0.0, 'pq': 0.0, 'overall': 0.0}] * len(pre_images)

def main():
    args = parse_args()
    
    metadata_path = os.path.join(args.results_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"Error: metadata.json not found in {args.results_dir}")
        return
        
    with open(metadata_path, 'r') as f:
        samples = json.load(f)
        
    print(f"Found {len(samples)} samples to evaluate.")
    
    evaluator = RSCCEvaluator(
        device=args.device,
        use_viescore=args.use_viescore,
        viescore_model=args.viescore_model
    )
    results = []
    
    images_dir = os.path.join(args.results_dir, "images")
    
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
                
            edited_img = Image.open(edited_path).convert("RGB")
            
            # Load Source and GT (paths are in metadata)
            pre_img = Image.open(sample['pre_image']).convert("RGB").resize(edited_img.size)
            gt_img = Image.open(sample['post_image']).convert("RGB").resize(edited_img.size)
            
            loaded_samples.append({
                'sample': sample,
                'edited_img': edited_img,
                'pre_img': pre_img,
                'gt_img': gt_img,
                'instruction': sample['instruction']
            })
        except Exception as e:
            print(f"Error loading sample {sample.get('id')}: {e}")
            continue
    
    print(f"Computing metrics for {len(loaded_samples)} samples in batches of {args.batch_size}...")
    
    # Process in batches
    for batch_start in tqdm(range(0, len(loaded_samples), args.batch_size), desc="Computing metrics"):
        batch_end = min(batch_start + args.batch_size, len(loaded_samples))
        batch = loaded_samples[batch_start:batch_end]
        
        try:
            # Prepare batch data
            edited_imgs = [item['edited_img'] for item in batch]
            pre_imgs = [item['pre_img'] for item in batch]
            gt_imgs = [item['gt_img'] for item in batch]
            instructions = [item['instruction'] for item in batch]
            
            # Compute metrics in batch
            dgtrs_scores = evaluator.compute_dgtrs_score_batch(edited_imgs, instructions)
            dinov3_scores = evaluator.compute_dinov3_similarity_batch(pre_imgs, edited_imgs)
            lpips_scores = evaluator.compute_lpips_batch(edited_imgs, gt_imgs)
            
            # Compute VIEScore if enabled (processes sequentially via API)
            if args.use_viescore and evaluator.viescore_evaluator is not None:
                viescore_results = evaluator.compute_viescore_batch(pre_imgs, edited_imgs, instructions)
            else:
                viescore_results = [{'sc': 0.0, 'pq': 0.0, 'overall': 0.0}] * len(batch)
            
            # Store results
            for i, item in enumerate(batch):
                item['sample']['metrics'] = {
                    'dgtrs_clip': dgtrs_scores[i],
                    'dinov3_sat': dinov3_scores[i],
                    'lpips': lpips_scores[i]
                }
                
                # Add VIEScore metrics if enabled
                if args.use_viescore:
                    item['sample']['metrics']['viescore_sc'] = viescore_results[i]['sc']
                    item['sample']['metrics']['viescore_pq'] = viescore_results[i]['pq']
                    item['sample']['metrics']['viescore_overall'] = viescore_results[i]['overall']
                
                results.append(item['sample'])
                
        except Exception as e:
            print(f"Error evaluating batch starting at {batch_start}: {e}")
            # Fallback to individual processing for this batch
            for item in batch:
                try:
                    dgtrs_score = evaluator.compute_dgtrs_score(item['edited_img'], item['instruction'])
                    dinov3_score = evaluator.compute_dinov3_similarity(item['pre_img'], item['edited_img'])
                    lpips_score = evaluator.compute_lpips(item['edited_img'], item['gt_img'])
                    
                    item['sample']['metrics'] = {
                        'dgtrs_clip': dgtrs_score,
                        'dinov3_sat': dinov3_score,
                        'lpips': lpips_score
                    }
                    
                    # Add VIEScore if enabled
                    if args.use_viescore and evaluator.viescore_evaluator is not None:
                        viescore_result = evaluator.compute_viescore(
                            item['pre_img'], item['edited_img'], item['instruction']
                        )
                        item['sample']['metrics']['viescore_sc'] = viescore_result['sc']
                        item['sample']['metrics']['viescore_pq'] = viescore_result['pq']
                        item['sample']['metrics']['viescore_overall'] = viescore_result['overall']
                    
                    results.append(item['sample'])
                except Exception as e2:
                    print(f"Error evaluating sample {item['sample'].get('id')}: {e2}")
                    continue

    if not results:
        print("No results generated.")
        return

    metrics = ['dgtrs_clip', 'dinov3_sat', 'lpips']
    
    # Add VIEScore metrics if enabled
    if args.use_viescore:
        metrics.extend(['viescore_sc', 'viescore_pq', 'viescore_overall'])
    
    avg_scores = {m: np.mean([r['metrics'][m] for r in results]) for m in metrics}
    
    print("\nBenchmark Results:")
    for m, score in avg_scores.items():
        print(f"{m}: {score:.4f}")
        
    # Save detailed metrics
    out_file = os.path.join(args.results_dir, "evaluation_metrics.json")
    with open(out_file, 'w') as f:
        json.dump({'average': avg_scores, 'details': results}, f, indent=2)
    print(f"Metrics saved to {out_file}")

if __name__ == "__main__":
    # Example CLI usage (quick demo):
    # python evaluate.py --results_dir /path/to/results --device cuda --batch_size 8
    # 
    # With VIEScore enabled:
    # python evaluate.py --results_dir /path/to/results --device cuda --batch_size 8 --use_viescore
    main()

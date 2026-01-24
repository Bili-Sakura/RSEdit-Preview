# RSEdit: Text-Guided Image Editing for Remote Sensing

RSEdit is a framework for instruction-based remote sensing image editing given text-to-image diffusion models. It propose a universal framework for both U-Net based (e.g. Stable Diffusion) and DiT based (e.g. Pixart-α, Stable Diffusion 3).

## 🛠️ Installation

We are working on python 3.12 with cuda 12.6 and PyTorch 2.5, but mostly, other version can work as well.

```bash
# install core ML packages with specific version
# see details in https://pytorch.org/get-started/locally/
pip install torch==2.5.0+cu124 torchaudio==2.5.0+cu124 torchvision==0.20.0+cu124 --index-url https://download.pytorch.org/whl/cu124
# then install all other libs
pip install -r rsedit.txt
# for evaluation/benchmark, additional install ever and torchange engine
# see details in https://github.com/Z-Zheng/pytorch-change-models and https://github.com/Z-Zheng/ever/
pip install torchange ever-beta
```

```bash
# (Optional but recommenced) Install xformers for faster training/inference:
pip install xformers==0.0.28.post2 -index-url https://download.pytorch.org/whl/cu124
# (Optional but recommenced) Install Flash Attention 2 for faster training/inference:
# wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install flash-attn --no-build-isolation
```

## 🚀 Quick Start

To generate an edited image using a pre-trained RSEdit model, follow the examples below for DiT or UNet-based architectures.

<details>
<summary><strong>RSEdit-DiT <a href="https://huggingface.co/BiliSakura/RSEdit-DiT" target="_blank">[Model Checkpoint]</a></strong></summary>

<br/>

The DiT-based model uses a custom pipeline to concatenate source image tokens with noisy latents.

```python
import torch
from PIL import Image
from diffusers import DiffusionPipeline

# Load model with custom pipeline
model_path = "BiliSakura/RSEdit-DiT" # or local path
pipe = DiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    custom_pipeline="pipeline_rsedit_dit.py"
).to("cuda")

# Switch to AttnProcessor (required for RSEdit DiT)
from diffusers.models.attention_processor import AttnProcessor
pipe.transformer.set_attn_processor(AttnProcessor())

# Load source image and edit
source_image = Image.open("satellite_image.png").convert("RGB")
prompt = "Flood the coastal area"
edited_image = pipe(
    prompt=prompt,
    source_image=source_image,
    num_inference_steps=50,
    guidance_scale=4.5,
    height=512,
    width=512,
).images[0]
edited_image.save("result_dit.png")
```

</details>

<details>
<summary><strong>RSEdit-UNet <a href="https://huggingface.co/BiliSakura/RSEdit-UNet-text-ablation" target="_blank">[Model Checkpoint]</a></strong></summary>

<br/>

The UNet-based models use the standard InstructPix2Pix pipeline.

```python
import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel

checkpoint_path = "BiliSakura/RSEdit-UNet-text-ablation/DGTRS-CLIP-ViT-L-14"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    checkpoint_path,
    torch_dtype=torch.bfloat16,
    safety_checker=None,
)

pipe = pipe.to("cuda")

# Load source image and edit
source_image = Image.open("satellite_image.png").convert("RGB")
prompt = "Add new residential blocks"
edited_image = pipe(
    prompt=prompt,
    image=source_image,
    num_inference_steps=50,
    guidance_scale=7.5,
    image_guidance_scale=1.5,
).images[0]
edited_image.save("result_unet.png")
```

</details>

## 🏋️ Training

RSEdit supports training on various architectures including SD3, DiT, and UNet.

### Prepare Dataset

We are using `RSCC-RSEdit-Test-Split` on [Hugging Face Dataset](https://huggingface.co/datasets/BiliSakura/RSCC-RSEdit-Test-Split). We can also formulate your own dataset with a (input image, edited image, edit prompt) tuple.

### Launch Training

You can use the provided script to start training:

```bash
# Edit src/rsedit-light/train.sh with your paths and configurations
bash src/rsedit-light/train.sh
```

Or launch manually with `accelerate`:

```bash
accelerate launch src/rsedit-light/train_sd3.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-3-medium-diffusers" \
  --train_data_dir="/path/to/dataset" \
  --resolution=512 \
  --train_batch_size=2 \
  --use_prodigy \
  --output_dir="checkpoints/rsedit-sd3"
```

## 🔍 Inference & Generation

Use `benchmark/generate.py` to run batch inference on test datasets (RSCC, LevirCC, SECOND-CC):

```bash
python benchmark/generate.py --model_name rsedit-dit --dataset_type rscc --device 0
```

Supported models see `benchmark/generate.py` for the full list.

## 📊 Benchmark

> See our results in https://huggingface.co/datasets/BiliSakura/RSEdit-Benchmark-Results.

Evaluate your results using the benchmark script which supports DGTRS-CLIP, DINOv3-Sat, LPIPS, and VIEScore.

```bash
python benchmark/evaluate.py --results_dir /path/to/results --device cuda --use_viescore
```

*Note: VIEScore requires an OpenRouter API key.*

## 📜 Citation (preprint paper coming soon)

If you find this work useful, please cite:

```bibtex
@misc{rsedit2026,
  author       = {Zhenyuan Chen, Zechuan Zhang, Feng Zhang},
  title        = {RSEdit: Text-Guided Image Editing for Remote Sensing},
  howpublished = {\url{https://github.com/Bili-Sakura/RSEdit-preview}},
  year         = {2026}
}
```

## 🙏 Acknowledgments

This project builds upon several amazing open-source libraries and models:

- [Diffusers](https://github.com/huggingface/diffusers)
- [Accelerate](https://github.com/huggingface/accelerate)
- [Stable Diffusion](https://github.com/Stability-AI/generative-models)
- [PixArt-Alpha](https://github.com/PixArt-alpha/PixArt-alpha)
- [Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium)


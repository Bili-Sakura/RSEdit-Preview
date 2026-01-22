#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The RSEdit Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
RSEdit SD3 Pipeline that adapts Stable Diffusion 3 (MMDiT) for instruction-based
image editing via Token Concatenation.

We concatenate source-image latents with noisy target latents along the width
dimension, letting the SD3 transformer learn in-context editing similarly to the
DiT-based PixArt variant.
"""

from typing import Any, Callable, Dict, List, Optional, Union

import PIL.Image
import torch
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from diffusers import StableDiffusion3Pipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import SD3IPAdapterMixin, SD3LoraLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import SD3Transformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    replace_example_docstring,
)
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from PIL import Image
        >>> from pipeline_rsedit_sd3 import RSEditSD3Pipeline
        >>> pipe = RSEditSD3Pipeline.from_pretrained(
        ...     "path/to/rsedit-sd3-model",
        ...     torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")
        >>> source = Image.open("satellite.png").convert("RGB").resize((1024, 1024))
        >>> prompt = "Add new residential blocks on the western side"
        >>> image = pipe(
        ...     prompt=prompt,
        ...     source_image=source,
        ...     num_inference_steps=28,
        ...     guidance_scale=5.0,
        ... ).images[0]
        >>> image.save("edited.png")
        ```
"""


class RSEditSD3Pipeline(StableDiffusion3Pipeline, SD3LoraLoaderMixin, SD3IPAdapterMixin):
    r"""
    Pipeline for instruction-based image editing using Stable Diffusion 3 with Token Concatenation.

    This pipeline adapts Stable Diffusion 3 (MMDiT) for image editing by concatenating source-image
    latents with noisy target latents along the width dimension, enabling in-context editing similar
    to the DiT-based PixArt variant.

    Args:
        transformer ([`SD3Transformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModelWithProjection`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant,
            with an additional added projection layer that is initialized with a diagonal matrix with the `hidden_size`
            as its dimension.
        text_encoder_2 ([`CLIPTextModelWithProjection`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        text_encoder_3 ([`T5EncoderModel`]):
            Frozen text-encoder. Stable Diffusion 3 uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the
            [t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_3 (`T5TokenizerFast`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->text_encoder_3->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "pooled_prompt_embeds"]

    def __init__(
        self,
        transformer: SD3Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_2: CLIPTokenizer,
        text_encoder_3: T5EncoderModel,
        tokenizer_3: T5TokenizerFast,
    ):
        super().__init__(
            transformer=transformer,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            text_encoder_3=text_encoder_3,
            tokenizer_3=tokenizer_3,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        latent_channels = self.vae.config.latent_channels if getattr(self, "vae", None) else 16
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            vae_latent_channels=latent_channels,
        )
        self.default_sample_size = (
            self.transformer.config.sample_size if hasattr(self, "transformer") and self.transformer is not None else 128
        )
        self.patch_size = (
            self.transformer.config.patch_size if hasattr(self, "transformer") and self.transformer is not None else 2
        )

    def _encode_source_image(
        self,
        source_image: PIL.Image.Image,
        device: torch.device,
        dtype: torch.dtype,
        num_images_per_prompt: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Encode the source image into latent space, applying SD3 scaling/shift.
        """
        if height is not None and width is not None:
            source_image = source_image.resize((width, height), PIL.Image.LANCZOS)

        image = self.image_processor.preprocess(source_image)
        image = image.to(device=device, dtype=dtype)

        latents = self.vae.encode(image).latent_dist.mode()
        latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        if num_images_per_prompt > 1:
            latents = latents.repeat(num_images_per_prompt, 1, 1, 1)
        return latents

    def _prepare_source_latents(
        self,
        source_image: Union[PIL.Image.Image, List[PIL.Image.Image]],
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        num_images_per_prompt: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        if isinstance(source_image, list):
            if len(source_image) != batch_size:
                raise ValueError(
                    f"Number of source images ({len(source_image)}) must match batch size ({batch_size})."
                )
            latents_list = [
                self._encode_source_image(img, device, dtype, num_images_per_prompt, height, width)
                for img in source_image
            ]
            source_latents = torch.cat(latents_list, dim=0)
        else:
            source_latents = self._encode_source_image(
                source_image, device, dtype, num_images_per_prompt, height, width
            )

        if batch_size > 1 and source_latents.shape[0] == 1:
            source_latents = source_latents.repeat(batch_size * num_images_per_prompt, 1, 1, 1)
        return source_latents

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        source_image: Union[PIL.Image.Image, List[PIL.Image.Image]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = "",
        num_inference_steps: int = 28,
        guidance_scale: float = 5.0,
        image_guidance_scale: Optional[float] = None,
        num_images_per_prompt: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        max_sequence_length: int = 256,
        clip_skip: Optional[int] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        mu: Optional[float] = None,
    ) -> Union[ImagePipelineOutput, tuple]:
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            source_image (`PIL.Image.Image` or `List[PIL.Image.Image]`):
                Source image(s) to edit. Required for RSEdit.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_inference_steps (`int`, *optional*, defaults to 28):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            image_guidance_scale (`float`, *optional*):
                Image guidance scale. If not provided, defaults to `guidance_scale`.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            height (`int`, *optional*):
                The height in pixels of the generated image.
            width (`int`, *optional*):
                The width in pixels of the generated image.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative pooled text embeddings will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            max_sequence_length (`int`, *optional*, defaults to 256):
                Maximum sequence length for text encoder.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            mu (`float`, *optional*):
                Mixing parameter for RSEdit editing.

        Returns:
            [`~pipelines.stable_diffusion.ImagePipelineOutput`] or `tuple`:
                [`~pipelines.stable_diffusion.ImagePipelineOutput`] if `return_dict` is True, otherwise a `tuple`.
                When returning a tuple, the first element is a list with the generated images.

        Examples:
        """
        if source_image is None:
            raise ValueError("`source_image` must be provided for RSEdit image editing.")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Either `prompt` or `prompt_embeds` must be provided.")

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        if image_guidance_scale is None:
            image_guidance_scale = guidance_scale
        do_classifier_free_guidance = guidance_scale > 1.0 and image_guidance_scale >= 1.0

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
            negative_prompt_3=negative_prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            clip_skip=clip_skip,
            max_sequence_length=max_sequence_length,
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds, negative_prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat(
                [pooled_prompt_embeds, negative_pooled_prompt_embeds, negative_pooled_prompt_embeds], dim=0
            )

        source_latents = self._prepare_source_latents(
            source_image,
            device=device,
            dtype=self.vae.dtype,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            height=height,
            width=width,
        )

        # Prepare latents
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # Timesteps
        scheduler_kwargs: Dict[str, Any] = {}
        if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            image_seq_len = (height // self.transformer.config.patch_size) * (width // self.transformer.config.patch_size)
            mu = self._calculate_mu(image_seq_len)
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps=num_inference_steps, device=device, **scheduler_kwargs
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 3) if do_classifier_free_guidance else latents
                latent_model_input = latent_model_input.to(device=device)

                current_timestep = t.expand(latent_model_input.shape[0])

                if do_classifier_free_guidance:
                    source_latents_input = torch.cat(
                        [source_latents, source_latents, torch.zeros_like(source_latents)], dim=0
                    )
                else:
                    source_latents_input = source_latents

                concatenated_latents = torch.cat([source_latents_input, latent_model_input], dim=3)

                noise_pred = self.transformer(
                    hidden_states=concatenated_latents,
                    timestep=current_timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]

                target_width = latents.shape[3]
                noise_pred = noise_pred[:, :, :, target_width:]

                if noise_pred.shape[1] == 2 * num_channels_latents:
                    noise_pred, _ = noise_pred.chunk(2, dim=1)

                if do_classifier_free_guidance:
                    noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                    noise_pred = (
                        noise_pred_uncond
                        + guidance_scale * (noise_pred_text - noise_pred_image)
                        + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                    )

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

        # Decode
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        # Cast latents to VAE dtype to ensure compatibility (fixes bf16 training validation)
        image = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

    def _calculate_mu(self, image_seq_len: int) -> float:
        base_seq_len = self.scheduler.config.get("base_image_seq_len", 256)
        max_seq_len = self.scheduler.config.get("max_image_seq_len", 4096)
        base_shift = self.scheduler.config.get("base_shift", 0.5)
        max_shift = self.scheduler.config.get("max_shift", 1.15)
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        return image_seq_len * m + b



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
RSEdit DiT Pipeline for Remote Sensing Image Editing using Token Concatenation.

This pipeline extends PixArtAlphaPipeline to support image-to-image editing
by concatenating source image tokens with noisy latent tokens, allowing the
DiT to leverage its sequence modeling capabilities for instruction-based editing.
"""

from typing import Callable, List, Optional, Union

import PIL.Image
import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer

from diffusers import PixArtAlphaPipeline, AutoencoderKL, PixArtTransformer2DModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from PIL import Image
        >>> from pipeline_rsedit_dit import RSEditDiTPipeline

        >>> # Load pipeline
        >>> pipe = RSEditDiTPipeline.from_pretrained(
        ...     "path/to/rsedit-dit-model",
        ...     torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> # Load source satellite image
        >>> source_image = Image.open("satellite_image.png").convert("RGB")
        
        >>> # Edit with instruction
        >>> prompt = "Flood the coastal area"
        >>> edited_image = pipe(
        ...     prompt=prompt,
        ...     source_image=source_image,
        ...     num_inference_steps=50,
        ...     guidance_scale=4.5,
        ... ).images[0]
        
        >>> edited_image.save("flooded_coastal_area.png")
        ```
"""


class RSEditDiTPipeline(PixArtAlphaPipeline):
    """
    Pipeline for RSEdit: Remote Sensing Image Editing using DiT with Token Concatenation.
    
    This pipeline extends PixArtAlphaPipeline to support instruction-based image editing
    for satellite imagery. It uses the Token Concatenation strategy where source image
    latents are concatenated with noisy target latents along the spatial width dimension,
    allowing the transformer to perform in-context learning for image-to-image translation.
    
    The pipeline uses the following components:
        - PixArtTransformer2DModel: Diffusion Transformer for denoising
        - T5EncoderModel: Text encoder for instruction embeddings
        - AutoencoderKL: VAE for encoding/decoding images to/from latent space
        
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder. PixArt-Alpha uses T5.
        tokenizer ([`T5Tokenizer`]):
            Tokenizer of class T5Tokenizer.
        transformer ([`PixArtTransformer2DModel`]):
            A PixArt transformer to denoise the encoded image latents.
        scheduler ([`KarrasDiffusionSchedulers`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: T5EncoderModel,
        tokenizer: T5Tokenizer,
        transformer: PixArtTransformer2DModel,
        scheduler: KarrasDiffusionSchedulers,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
        
    def _encode_source_image(
        self,
        source_image: PIL.Image.Image,
        device: torch.device,
        dtype: torch.dtype,
        num_images_per_prompt: int = 1,
    ) -> torch.Tensor:
        """
        Encode the source image into latent space.
        
        Args:
            source_image: PIL Image to encode
            device: Device to place the latents on
            dtype: Data type for the latents (used for output, VAE uses its own dtype)
            num_images_per_prompt: Number of images to generate per prompt
            
        Returns:
            Encoded latents of shape (batch_size * num_images_per_prompt, channels, height, width)
        """
        # Convert PIL image to tensor
        image_np = np.array(source_image.convert("RGB")).astype(np.float32) / 127.5 - 1.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
        # Use VAE's dtype for encoding to ensure compatibility with mixed precision
        image_tensor = image_tensor.to(device=device, dtype=self.vae.dtype)
        
        # Encode to latent space (use mode for deterministic encoding)
        latents = self.vae.encode(image_tensor).latent_dist.mode()
        latents = latents * self.vae.config.scaling_factor
        
        # Ensure latents are on the correct device (critical for multi-GPU with device_map)
        # The VAE encoder might be on a different GPU, so we need to move the output
        latents = latents.to(device=device)
        
        # Cast back to requested dtype for pipeline consistency
        latents = latents.to(dtype=dtype)
        
        # Duplicate for num_images_per_prompt
        if num_images_per_prompt > 1:
            latents = latents.repeat(num_images_per_prompt, 1, 1, 1)
            
        return latents

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        source_image: Union[PIL.Image.Image, List[PIL.Image.Image]] = None,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 4.5,
        image_guidance_scale: Optional[float] = None,
        num_images_per_prompt: Optional[int] = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        clean_caption: bool = True,
        use_resolution_binning: bool = True,
        max_sequence_length: int = 120,
        **kwargs,
    ) -> Union[ImagePipelineOutput, tuple]:
        """
        Function invoked when calling the pipeline for generation.
        
        Args:
            prompt (`str` or `List[str]`, *optional*):
                The editing instruction prompt or prompts to guide image generation. If not defined, you need
                to pass `prompt_embeds`.
            source_image (`PIL.Image.Image` or `List[PIL.Image.Image]`):
                The source satellite image(s) to edit.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process. If not defined, equal spaced `num_inference_steps`
                timesteps are used.
            guidance_scale (`float`, *optional*, defaults to 4.5):
                Guidance scale as defined in Classifier-Free Guidance (CFG).
                Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            image_guidance_scale (`float`, *optional*):
                Image guidance scale for controlling the influence of the source image. If None, uses `guidance_scale`.
                This allows separate control over text vs. image conditioning strength.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            height (`int`, *optional*, defaults to self.transformer.config.sample_size):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.transformer.config.sample_size):
                The width in pixels of the generated image.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper. Only applies to DDIMScheduler.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of torch generator(s) to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
            prompt_attention_mask (`torch.FloatTensor`, *optional*):
                Pre-generated attention mask for text embeddings.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings.
            negative_prompt_attention_mask (`torch.FloatTensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                `"pil"` (`PIL.Image.Image`), `"np"` (`np.array`) or `"pt"` (`torch.Tensor`).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called.
            clean_caption (`bool`, *optional*, defaults to `True`):
                Whether or not to clean the caption before creating embeddings.
            use_resolution_binning (`bool`, *optional*, defaults to `True`):
                Whether to use resolution binning for PixArt models.
            max_sequence_length (`int`, *optional*, defaults to 120):
                Maximum sequence length for text encoder.
                
        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images.

        Examples:
        """
        # 1. Check inputs
        if source_image is None:
            raise ValueError("`source_image` must be provided for RSEdit image editing.")
            
        if prompt is None and prompt_embeds is None:
            raise ValueError("Either `prompt` or `prompt_embeds` must be provided.")
        
        if height is None:
            height = self.transformer.config.sample_size * self.vae_scale_factor
        if width is None:
            width = self.transformer.config.sample_size * self.vae_scale_factor

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        
        # 3. Encode source image
        if isinstance(source_image, PIL.Image.Image):
            source_image = source_image.resize((width, height), PIL.Image.LANCZOS)
        elif isinstance(source_image, list):
            source_image = [img.resize((width, height), PIL.Image.LANCZOS) for img in source_image]
            if len(source_image) != batch_size:
                raise ValueError(
                    f"Number of source images ({len(source_image)}) must match batch size ({batch_size})"
                )
        
        # Encode source image(s)
        if isinstance(source_image, list):
            source_latents_list = []
            for img in source_image:
                latent = self._encode_source_image(
                    img, device, self.vae.dtype, num_images_per_prompt
                )
                # Ensure latent is on the correct device (critical for multi-GPU)
                latent = latent.to(device=device)
                source_latents_list.append(latent)
            source_latents = torch.cat(source_latents_list, dim=0)
        else:
            source_latents = self._encode_source_image(
                source_image, device, self.vae.dtype, num_images_per_prompt
            )
        
        # Duplicate source latents for batch
        if batch_size > 1 and source_latents.shape[0] == 1:
            source_latents = source_latents.repeat(batch_size * num_images_per_prompt, 1, 1, 1)

        # 4. Encode input prompt
        # Default image_guidance_scale to guidance_scale if not provided
        if image_guidance_scale is None:
            image_guidance_scale = guidance_scale
        
        do_classifier_free_guidance = guidance_scale > 1.0 and image_guidance_scale >= 1.0
        
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt,
            do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            clean_caption=clean_caption,
            max_sequence_length=max_sequence_length,
        )
        
        if do_classifier_free_guidance:
            # For InstructPix2Pix: [text_embeds, negative_embeds, negative_embeds]
            # Corresponds to: [Text+Image, Image-Only, Unconditional]
            # But standard diffusers usually does [neg, pos]. 
            # We need 3 components for IP2P: (Text+Image), (Image), (None)
            
            # Re-arranging to match: [Text+Image, Image, None]
            # prompt_embeds contains the "positive" text.
            # negative_prompt_embeds contains the "negative/null" text.
            
            # Batch structure: [Positive Text, Negative Text, Negative Text]
            prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds, negative_prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([prompt_attention_mask, negative_prompt_attention_mask, negative_prompt_attention_mask], dim=0)

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
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
        
        # Ensure source_latents are on the same device as latents (critical for multi-GPU)
        # This ensures all concatenations in the denoising loop work correctly
        source_latents = source_latents.to(device=latents.device)

        # 7. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Prepare added time ids & embeddings (resolution, aspect ratio)
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        if self.transformer.config.sample_size == 128:
            resolution = torch.tensor([height, width]).repeat(batch_size * num_images_per_prompt, 1)
            aspect_ratio = torch.tensor([float(height / width)]).repeat(batch_size * num_images_per_prompt, 1)
            resolution = resolution.to(dtype=prompt_embeds.dtype, device=device)
            aspect_ratio = aspect_ratio.to(dtype=prompt_embeds.dtype, device=device)
            
            if do_classifier_free_guidance:
                resolution = torch.cat([resolution, resolution, resolution], dim=0)
                aspect_ratio = torch.cat([aspect_ratio, aspect_ratio, aspect_ratio], dim=0)
            
            added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

        # 9. Denoising loop with Token Concatenation
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand latents for classifier-free guidance
                # IP2P: 3 copies [latents, latents, latents]
                latent_model_input = torch.cat([latents] * 3) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # **Token Concatenation Strategy**
                # Concatenate source image latents with noisy target latents along width dimension
                # IP2P Batch: [Source, Source, Zero_Source]
                if do_classifier_free_guidance:
                    source_latents_input = torch.cat([source_latents, source_latents, torch.zeros_like(source_latents)], dim=0)
                else:
                    source_latents_input = source_latents
                
                # Ensure both tensors are on the same device before concatenation (critical for multi-GPU)
                source_latents_input = source_latents_input.to(device=latent_model_input.device)
                concatenated_latents = torch.cat([source_latents_input, latent_model_input], dim=3)
                
                # Expand the timesteps for the expanded latents (CFG)
                current_timestep = t
                if not torch.is_tensor(current_timestep):
                    is_mps = concatenated_latents.device.type == "mps"
                    is_npu = concatenated_latents.device.type == "npu"
                    if isinstance(current_timestep, float):
                        dtype = torch.float32 if (is_mps or is_npu) else torch.float64
                    else:
                        dtype = torch.int32 if (is_mps or is_npu) else torch.int64
                    current_timestep = torch.tensor([current_timestep], dtype=dtype, device=concatenated_latents.device)
                elif len(current_timestep.shape) == 0:
                    current_timestep = current_timestep[None].to(concatenated_latents.device)
                
                current_timestep = current_timestep.expand(concatenated_latents.shape[0])

                # Predict noise residual
                noise_pred = self.transformer(
                    concatenated_latents,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    timestep=current_timestep,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # **Extract target portion** (right half corresponding to edited image)
                # The model predicts noise for both source and target, we only need target
                target_width = latents.shape[3]
                noise_pred = noise_pred[:, :, :, target_width:]

                # Split model prediction if it contains variance (PixArt can output 8 channels)
                if noise_pred.shape[1] == 2 * num_channels_latents:
                    noise_pred, _ = noise_pred.chunk(2, dim=1)

                # Perform classifier-free guidance
                if do_classifier_free_guidance:
                    # noise_pred batch: [Text+Image, Image, None]
                    noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                    
                    # IP2P CFG Formula:
                    # pred = uncond + s_text * (text - image) + s_image * (image - uncond)
                    # Note: s_text is usually just `guidance_scale`. s_image is `image_guidance_scale`.
                    
                    # But wait, standard CFG is: uncond + s * (cond - uncond)
                    # IP2P paper eq: e_theta = e(phi, phi) + s_T * (e(c_I, c_T) - e(c_I, phi)) + s_I * (e(c_I, phi) - e(phi, phi))
                    # Mapping:
                    # e(c_I, c_T) -> noise_pred_text (Full)
                    # e(c_I, phi) -> noise_pred_image (Image only, Null Text)
                    # e(phi, phi) -> noise_pred_uncond (Unconditional)
                    
                    noise_pred = (
                        noise_pred_uncond
                        + guidance_scale * (noise_pred_text - noise_pred_image)
                        + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                    )

                # Compute previous noisy sample: x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # Call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # 10. Post-processing
        if not output_type == "latent":
            # Cast latents to VAE dtype to ensure compatibility (fixes bf16 training validation)
            image = self.vae.decode(latents.to(self.vae.dtype) / self.vae.config.scaling_factor, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
        else:
            image = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


"""
VIEScore integration for RSEdit using OpenRouter or DashScope API

This module provides VIEScore metric evaluation using vision-language models
(e.g., google/gemini-3-flash-preview, qwen-vl-max).
"""

import os
import json
import math
import base64
import requests
from typing import List, Dict, Any, Optional, Union
from PIL import Image
from io import BytesIO


# =============================================================================
# API Configuration
# =============================================================================

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DASHSCOPE_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


def get_api_key(name: str) -> str:
    """
    Get API key from environment variable or .env file.
    
    Checks in order:
    1. /path/to/.env
    2. Environment variable {name}
    3. .env file at ./path/to/.env (relative to script directory)
    """
    # 1. Try explicit path first
    explicit_env = "/path/to/.env"
    
    # Try loading from .env files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rel_env = os.path.normpath(os.path.join(script_dir, "..", ".env"))
    
    for env_file in [explicit_env, rel_env]:
        if os.path.exists(env_file):
            try:
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        # Skip comments and empty lines
                        if not line or line.startswith('#'):
                            continue
                        # Parse KEY=VALUE format
                        if '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            # Remove quotes if present
                            if value.startswith('"') and value.endswith('"'):
                                value = value[1:-1]
                            elif value.startswith("'") and value.endswith("'"):
                                value = value[1:-1]
                            
                            if key == name:
                                return value
            except Exception as e:
                print(f"Warning: Could not read .env file {env_file}: {e}")
    
    # 2. Check environment variable
    api_key = os.environ.get(name)
    if api_key:
        return api_key
    
    # If still not found, return None
    return None


def get_openrouter_api_key() -> str:
    api_key = get_api_key("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found.")
    return api_key


# =============================================================================
# VIEScore Prompts (from VIEScore project)
# =============================================================================

_context_no_delimit = """You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

You will have to give your output in this way (Keep your reasoning concise and short.):
{
"score" : [...],
"reasoning" : "..."
}"""

_prompts_0shot_two_image_edit_rule = """RULES:

Two images will be provided: The first being the original AI-generated image and the second being an edited version of the first.
The objective is to evaluate how successfully the editing instruction has been executed in the second image.

Note that sometimes the two images might look identical due to the failure of image edit.
"""

_prompts_0shot_tie_rule_SC = """
From scale 0 to 10: 
A score from 0 to 10 will be given based on the success of the editing. (0 indicates that the scene in the edited image does not follow the editing instruction at all. 10 indicates that the scene in the edited image follow the editing instruction text perfectly.)
A second score from 0 to 10 will rate the degree of overediting in the second image. (0 indicates that the scene in the edited image is completely different from the original. 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the editing success and 'score2' evaluates the degree of overediting.

Editing instruction: <instruction>
"""

_prompts_0shot_rule_PQ = """RULES:

The image is an AI-generated image.
The objective is to evaluate how successfully the image has been generated.

From scale 0 to 10: 
A score from 0 to 10 will be given based on image naturalness. 
(
    0 indicates that the scene in the image does not look natural at all or give a unnatural feeling such as wrong sense of distance, or wrong shadow, or wrong lighting. 
    10 indicates that the image looks natural.
)
A second score from 0 to 10 will rate the image artifacts. 
(
    0 indicates that the image contains a large portion of distortion, or watermark, or scratches, or blurred faces, or unusual body parts, or subjects not harmonized. 
    10 indicates the image has no artifacts.
)
Put the score in a list such that output score = [naturalness, artifacts]
"""


# =============================================================================
# Utility Functions
# =============================================================================

def pil_image_to_base64(pil_image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    pil_image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


def mllm_output_to_dict(input_string: str, give_up_parsing: bool = False) -> Union[Dict[str, Any], bool]:
    """
    Parse MLLM output string to dictionary.
    Adapted from VIEScore utils.py
    """
    import re
    import random
    
    if input_string == "rate_limit_exceeded":
        return "rate_limit_exceeded"
    
    # Try to find JSON content
    start_index = input_string.find('{')
    end_index = input_string.rfind('}') + 1
    
    if start_index == -1 or end_index == 0:
        # JSON not found, try to find a list
        start_index = input_string.find('[')
        end_index = input_string.rfind(']') + 1
        
        if give_up_parsing:
            guessed_value = random.randint(0, 10)
            print(f"Failed to parse output. Guessing value: {guessed_value}")
            return {'score': [guessed_value], "reasoning": f"guess_if_cannot_parse | {input_string}"}
        
        if start_index != -1 and end_index != 0:
            try:
                scores = json.loads(input_string[start_index:end_index])
                if not isinstance(scores, list):
                    scores = [scores]
                return {'score': scores, "reasoning": "System: output is a list of scores"}
            except:
                pass
        
        print("Failed to find JSON content in output.")
        return False
    
    # Extract and parse JSON
    json_str = input_string[start_index:end_index].strip()
    json_str = json_str.replace("\n", "")
    
    try:
        new_data = json.loads(json_str)
        if not isinstance(new_data['score'], list):
            new_data['score'] = [new_data['score']]
        return new_data
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"JSON string: {json_str}")
        return False


# =============================================================================
# API Adapters
# =============================================================================

class OpenAICompatibleAdapter:
    """
    Vision-language model adapter for OpenAI-compatible APIs (OpenRouter, DashScope, etc.).
    """
    
    def __init__(
        self, 
        model_name: str, 
        api_key: str, 
        api_url: str,
        temperature: float = 1.0,
        max_tokens: int = 4096
    ):
        """
        Initialize the adapter.
        
        Args:
            model_name: Model name to use
            api_key: API key
            api_url: API endpoint URL
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.model_name = model_name
        self.api_key = api_key
        self.api_url = api_url
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def prepare_prompt(self, image_links: List[Union[str, Image.Image]], text_prompt: str) -> List:
        """
        Prepare prompt for the API.
        
        Args:
            image_links: List of image paths or PIL Images
            text_prompt: Text prompt
            
        Returns:
            List containing [images, text_prompt]
        """
        if not isinstance(image_links, list):
            image_links = [image_links]
        
        # Convert all images to PIL Images
        images = []
        for img in image_links:
            if isinstance(img, str):
                images.append(Image.open(img).convert("RGB"))
            elif isinstance(img, Image.Image):
                images.append(img.convert("RGB"))
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
        
        return [images, text_prompt]
    
    def get_parsed_output(self, prompt: List) -> str:
        """
        Get output from the API.
        
        Args:
            prompt: Prepared prompt [images, text]
            
        Returns:
            Response text from the model
        """
        images = prompt[0]
        text_prompt = prompt[1]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Build message content with images
        content = []
        
        # Add images as base64
        for img in images:
            img_base64 = pil_image_to_base64(img, format="JPEG")
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}"
                }
            })
        
        # Add text prompt
        content.append({
            "type": "text",
            "text": text_prompt
        })
        
        # Build payload
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        # Make API request
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract text response
            if "choices" in result and len(result["choices"]) > 0:
                message = result["choices"][0].get("message", {})
                return message.get("content", "")
            else:
                return f"Error: No response from API. Full response: {result}"
                
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return "Error in API request"


# For backward compatibility
OpenRouterGemini = OpenAICompatibleAdapter


# =============================================================================
# VIEScore Evaluator
# =============================================================================

class VIEScoreEvaluator:
    """
    VIEScore evaluator for image editing tasks.
    
    This evaluator uses OpenRouter or DashScope API with vision-language models 
    to compute VIEScore metrics:
    - SC (Semantic Consistency): Measures how well the edit follows the instruction
    - PQ (Perceptual Quality): Measures naturalness and artifacts
    - O (Overall): Geometric mean of SC and PQ
    """
    
    def __init__(
        self,
        model_name: str = "google/gemini-3-flash-preview",
        api_key: Optional[str] = None,
        api_url: Optional[str] = None
    ):
        """
        Initialize VIEScore evaluator.
        
        Args:
            model_name: Model name (default: google/gemini-3-flash-preview)
            api_key: API key (if None, reads from environment)
            api_url: API URL (if None, detected from model_name)
        """
        self.model_name = model_name
        
        # Determine provider and API key
        if "qwen" in model_name.lower():
            provider = "dashscope"
            key_name = "DASHSCOPE_API_KEY"
            default_url = DASHSCOPE_API_URL
        else:
            provider = "openrouter"
            key_name = "OPENROUTER_API_KEY"
            default_url = OPENROUTER_API_URL
        
        effective_api_key = api_key or get_api_key(key_name)
        if not effective_api_key:
            raise ValueError(f"API key {key_name} not found for {provider}")
            
        effective_api_url = api_url or os.getenv("API_BASE_URL") or default_url
        
        self.model = OpenAICompatibleAdapter(
            model_name=model_name,
            api_key=effective_api_key,
            api_url=effective_api_url,
            temperature=0.01 if "qwen" in model_name.lower() else 1.0  # Follow autoeval.py for qwen
        )
        
        # Task is "tie" (two-image editing) for RSEdit
        self.task = "tie"
        
        # Build prompts
        self.context = _context_no_delimit
        self.SC_prompt = "\n".join([
            self.context,
            _prompts_0shot_two_image_edit_rule,
            _prompts_0shot_tie_rule_SC
        ])
        self.PQ_prompt = "\n".join([
            self.context,
            _prompts_0shot_rule_PQ
        ])
        
        print(f"VIEScore evaluator initialized with model: {model_name}")
    
    def compute_viescore(
        self,
        pre_image: Image.Image,
        edited_image: Image.Image,
        instruction: str
    ) -> Dict[str, float]:
        """
        Compute VIEScore for a single image editing sample.
        
        Args:
            pre_image: Original image (PIL Image)
            edited_image: Edited image (PIL Image)
            instruction: Editing instruction text
            
        Returns:
            Dictionary with keys: 'sc', 'pq', 'overall'
        """
        try:
            # Prepare prompts
            _SC_prompt = self.SC_prompt.replace("<instruction>", instruction)
            
            # Prepare inputs (for tie task: [pre_image, edited_image])
            SC_prompt_final = self.model.prepare_prompt([pre_image, edited_image], _SC_prompt)
            PQ_prompt_final = self.model.prepare_prompt([edited_image], self.PQ_prompt)
            
            # Get responses
            SC_dict = False
            PQ_dict = False
            tries = 0
            max_tries = 1
            
            while SC_dict is False or PQ_dict is False:
                tries += 1
                guess_if_cannot_parse = True if tries > max_tries else False
                
                result_SC = self.model.get_parsed_output(SC_prompt_final)
                result_PQ = self.model.get_parsed_output(PQ_prompt_final)
                
                SC_dict = mllm_output_to_dict(result_SC, give_up_parsing=guess_if_cannot_parse)
                PQ_dict = mllm_output_to_dict(result_PQ, give_up_parsing=guess_if_cannot_parse)
                
                if SC_dict == "rate_limit_exceeded" or PQ_dict == "rate_limit_exceeded":
                    raise ValueError("rate_limit_exceeded")
            
            # Extract scores
            SC_score = min(SC_dict['score'])
            PQ_score = min(PQ_dict['score'])
            O_score = math.sqrt(SC_score * PQ_score)
            
            return {
                'sc': SC_score,
                'pq': PQ_score,
                'overall': O_score
            }
            
        except Exception as e:
            print(f"Error computing VIEScore: {e}")
            return {
                'sc': 0.0,
                'pq': 0.0,
                'overall': 0.0
            }
    
    def compute_viescore_batch(
        self,
        pre_images: List[Image.Image],
        edited_images: List[Image.Image],
        instructions: List[str]
    ) -> List[Dict[str, float]]:
        """
        Compute VIEScore for a batch of image editing samples.
        
        Note: OpenRouter API processes samples sequentially, not in parallel.
        
        Args:
            pre_images: List of original images (PIL Images)
            edited_images: List of edited images (PIL Images)
            instructions: List of editing instruction texts
            
        Returns:
            List of dictionaries with keys: 'sc', 'pq', 'overall'
        """
        results = []
        for pre_img, edited_img, instruction in zip(pre_images, edited_images, instructions):
            result = self.compute_viescore(pre_img, edited_img, instruction)
            results.append(result)
        return results


# =============================================================================
# Test/Demo Code
# =============================================================================

if __name__ == "__main__":
    # Example usage
    print("Testing VIEScore evaluator with OpenRouter...")
    
    try:
        # Initialize evaluator
        evaluator = VIEScoreEvaluator()
        
        # Create dummy test images
        pre_image = Image.new("RGB", (512, 512), color="blue")
        edited_image = Image.new("RGB", (512, 512), color="red")
        instruction = "Change the color to red"
        
        # Compute score
        print(f"\nComputing VIEScore for test image...")
        scores = evaluator.compute_viescore(pre_image, edited_image, instruction)
        
        print(f"\nResults:")
        print(f"  SC (Semantic Consistency): {scores['sc']:.2f}")
        print(f"  PQ (Perceptual Quality): {scores['pq']:.2f}")
        print(f"  Overall: {scores['overall']:.2f}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure appropriate API key (OPENROUTER_API_KEY or DASHSCOPE_API_KEY) is set.")

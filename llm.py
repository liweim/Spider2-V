import requests
import os
import time
import threading
from functools import wraps
from PIL import Image
import json
import base64
import io
from dataclasses import dataclass
from typing import Any, Tuple, Optional
from configs.config import *
import openai
from openai import OpenAI

# Fix numpy import issue in CUDA environment
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"


@dataclass
class UsageStats:
    """Unified usage statistics data structure"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    image_count: int = 0
    
    def add(self, other: 'UsageStats'):
        """Accumulate statistics from another UsageStats object"""
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.image_count += other.image_count


@dataclass
class ModelConfig:
    """Complete configuration for a model including client and pricing"""
    client_class: str  # Name of the API client class
    real_model_name: str  # Actual model name used by the API
    is_vlm: bool # Whether the model is a VLM model
    prompt_price: float  # per 1M tokens
    completion_price: float  # per 1M tokens
    image_price: float  # per 1K images
    
    def calculate_cost(self, stats: UsageStats) -> float:
        """Calculate total cost based on usage statistics"""
        token_cost = (
            self.prompt_price * stats.prompt_tokens + 
            self.completion_price * stats.completion_tokens
        ) / 1_000_000
        image_cost = self.image_price * stats.image_count / 1_000
        return token_cost + image_cost


# Unified model configuration table
MODEL_CONFIGS = {
    "gpt-4o": ModelConfig("Road2allAPI", "gpt-4o-2024-05-13", True, 2.5, 10, 3.613),
    "gpt-o3": ModelConfig("Road2allAPI", "gpt-o3", True, 2, 8, 1.53),
    "gpt-o4-mini": ModelConfig("Road2allAPI", "gpt-o4-mini", True, 1.1, 4.4, 0.842),
    "computer-use-preview": ModelConfig("OpenAIAPI", "computer-use-preview", True, 3, 12, 0),
    "claude-3.5": ModelConfig("Road2allAPI", "claude-3-5-sonnet-20240620", True, 3, 15, 4.8),
    "claude-4": ModelConfig("Road2allAPI", "claude-sonnet-4-20250514", True, 3, 15, 4.8),
    "claude-4.5": ModelConfig("OpenRouterAPI", "anthropic/claude-sonnet-4.5", True, 3, 15, 4.8),
    "deepseek-v3": ModelConfig("Road2allAPI", "deepseek-v3", False, 0.24, 0.84, 0),
    "deepseek-r1": ModelConfig("Road2allAPI", "deepseek-reasoner", False, 0.4, 1.75, 0),
    "deepseek-v3.1": ModelConfig("OpenRouterAPI", "deepseek/deepseek-chat-v3.1:free", False, 0.2, 0.8, 0),
    "qwen2.5-vl-72b": ModelConfig("OpenRouterAPI", "qwen/qwen2.5-vl-72b-instruct", True, 0.07, 0.28, 0),
    "qwen3": ModelConfig("OpenRouterAPI", "qwen/qwen3-235b-a22b:free", False, 0, 0, 0),
    "gemini-2.5-pro": ModelConfig("Road2allAPI", "gemini-2.5-pro-preview-05-06", True, 1.25, 10, 5.16),
    "gemini-2.5-flash": ModelConfig("OpenRouterAPI", "google/gemini-2.5-flash", True, 0.3, 2.5, 1.238),
}


def resize_image(img, max_size=1024):
    """Resize image to meet API constraints"""
    width, height = img.size
    max_edge = max(width, height)

    if max_edge > max_size:
        ratio = max_size / max_edge
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        img = img.resize((new_width, new_height))
    return img


def encode_image(image: Image.Image) -> str:
    """Encode PIL image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_base64}"


def count_images_in_messages(messages: list) -> int:
    """Count the number of images in message list"""
    count = 0
    for message in messages:
        if isinstance(message.get("content"), list):
            for content in message["content"]:
                if content.get("type") == "image_url":
                    count += 1
    return count


def with_timeout(timeout_seconds):
    """Timeout decorator compatible with Windows"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout_seconds)

            if thread.is_alive():
                raise TimeoutError(
                    f"Function call timed out after {timeout_seconds} seconds"
                )

            if exception[0]:
                raise exception[0]

            return result[0]

        return wrapper
    return decorator


class BaseLLMClient:
    """Base class for LLM clients, defining unified interface"""
    
    def __init__(self, model_name: str, temperature: float = 0, max_tokens: int = 1024):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.usage_stats = UsageStats()
    
    def __call__(self, messages: list) -> str:
        """Send messages and return response"""
        raise NotImplementedError
    
    def call_cua(self, messages: list, **kwargs) -> Tuple[str, str]:
        """Computer Use API call (if supported)"""
        raise NotImplementedError("This model does not support Computer Use API")
    
    def get_usage_stats(self) -> UsageStats:
        """Get usage statistics"""
        return self.usage_stats
    
    def reset_stats(self):
        """Reset statistics"""
        self.usage_stats = UsageStats()


class OpenAIAPI(BaseLLMClient):
    """OpenAI API client"""
    
    def __init__(self, model_name: str, temperature: float = 0, max_tokens: int = 1024):
        super().__init__(model_name, temperature, max_tokens)
        self.client = OpenAI(api_key=OPENAI_API_KEY)
    
    def __call__(self, messages: list) -> str:
        response = self.client.responses.create(
            model=self.model_name,
            input=messages,
        )
        
        # Update statistics
        self.usage_stats.prompt_tokens += response.usage.input_tokens
        self.usage_stats.completion_tokens += response.usage.output_tokens
        self.usage_stats.image_count += count_images_in_messages(messages)
        
        return response.output[0].content[0].text
    
    def call_cua(
        self,
        messages: list,
        screen_width: int = 1920,
        screen_height: int = 1080,
        environment: str = "linux",
    ) -> Tuple[str, str]:
        """Call Computer Use API"""
        response = self.client.responses.create(
            model=self.model_name,
            tools=[{
                "type": "computer_use_preview",
                "display_width": screen_width,
                "display_height": screen_height,
                "environment": environment,
            }],
            input=messages,
            reasoning={"summary": "concise"},
            tool_choice="required",
            truncation="auto",
        )
        
        # Update statistics
        self.usage_stats.prompt_tokens += response.usage.input_tokens
        self.usage_stats.completion_tokens += response.usage.output_tokens
        self.usage_stats.image_count += 1
        
        reasoning = ""
        py_cmd = ""
        
        for output_item in response.output:
            output_type = output_item.get("type", "") if isinstance(output_item, dict) else getattr(output_item, "type", "")
            
            if "computer_call" in str(output_type):
                action_call = output_item if isinstance(output_item, dict) else output_item.model_dump()
                py_cmd = self._cua_to_pyautogui(action_call["action"])
                print(f"Executed: {py_cmd}")
                
            elif "reasoning" in str(output_type) and hasattr(output_item, 'summary') and len(output_item.summary) > 0:
                reasoning = output_item.summary[0].text
                print(f"Reasoning: {reasoning}")
                
            elif "message" in str(output_type):
                message_text = output_item.content[0].text if hasattr(output_item, 'content') else str(output_item)
                reasoning = message_text
        
        return py_cmd, reasoning
    
    def _cua_to_pyautogui(self, action) -> str:
        """Convert OpenAI CUA action to pyautogui command"""
        def fld(key: str, default: Any = None) -> Any:
            return action.get(key, default) if isinstance(action, dict) else getattr(action, key, default)

        act_type = fld("type")
        if not isinstance(act_type, str):
            act_type = str(act_type).split(".")[-1]
        act_type = act_type.lower()

        if act_type in ["click", "double_click"]:
            button = fld('button', 'left')
            if button in [1, 'left']:
                button = 'left'
            elif button in [2, 'middle']:
                button = 'middle'
            elif button in [3, 'right']:
                button = 'right'

            if act_type == "click":
                return f"pyautogui.click({fld('x')}, {fld('y')}, button='{button}')"
            if act_type == "double_click":
                return f"pyautogui.doubleClick({fld('x')}, {fld('y')}, button='{button}')"
            
        if act_type == "scroll":
            if fld('scroll_y', 0) != 0:
                return f"pyautogui.scroll({-fld('scroll_y', 0) / 100}, x={fld('x', 0)}, y={fld('y', 0)})"
            return ""
        
        if act_type == "drag":
            path = fld('path', [{"x": 0, "y": 0}, {"x": 0, "y": 0}])
            cmd = f"pyautogui.moveTo({path[0]['x']}, {path[0]['y']}, _pause=False); "
            cmd += f"pyautogui.dragTo({path[1]['x']}, {path[1]['y']}, duration=0.5, button='left')"
            return cmd

        if act_type == 'move':
            return f"pyautogui.moveTo({fld('x')}, {fld('y')})"

        if act_type == "keypress":
            keys = fld("keys", []) or [fld("key")]
            if len(keys) == 1:
                return f"pyautogui.press('{keys[0].lower()}')"
            else:
                return "pyautogui.hotkey('{}')".format("', '".join(keys)).lower()
            
        if act_type == "type":
            text = str(fld("text", ""))
            return "pyautogui.typewrite({:})".format(repr(text))
        
        return "WAIT"


class Road2allAPI(BaseLLMClient):
    """Road2all API client"""
    
    def __init__(self, model_name: str, temperature: float = 0, max_tokens: int = 1024):
        super().__init__(model_name, temperature, max_tokens)
        self.url = "https://api2.road2all.com/v1/chat/completions"
    
    def __call__(self, messages: list) -> str:
        headers = {
            "Authorization": f"Bearer {ROAD2ALL_API_KEY}",
            "Content-Type": "application/json",
        }
        
        data = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages,
        }
        
        response = requests.post(self.url, headers=headers, json=data)
        
        if response.status_code == 200:
            result_json = response.json()
            result = result_json["choices"][0]["message"]["content"]
            usage = result_json["usage"]
            
            # Update statistics
            self.usage_stats.prompt_tokens += usage["prompt_tokens"]
            self.usage_stats.completion_tokens += usage["completion_tokens"]
            self.usage_stats.image_count += count_images_in_messages(messages)
            
            return result
        else:
            raise Exception(response.json()["error"]["message"])
    
    def call_cua(self, messages: list, **kwargs) -> Tuple[str, str]:
        retry = 3
        for _ in range(retry):
            py_cmd = self.__call__(messages)
            if "```python" in py_cmd:
                py_cmd = py_cmd.split("```python")[1].split("```")[0]
                break
            else:
                print("Invalid response format, retrying: ", py_cmd)
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "You must return a python code within ```python``` code block to execute the task."},
                    ],
                })
        return py_cmd, ""


class OpenRouterAPI(BaseLLMClient):
    """OpenRouter API client"""
    
    def __init__(self, model_name: str, temperature: float = 0, max_tokens: int = 1024):
        super().__init__(model_name, temperature, max_tokens)
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
    
    def __call__(self, messages: list) -> str:
        data = json.dumps({
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": messages,
        })
        
        response = requests.post(self.url, headers=self.headers, data=data)
        
        if response.status_code == 200:
            result_json = response.json()
            if "choices" not in result_json:
                print(result_json)
                raise Exception("Invalid response format")
            
            result = result_json["choices"][0]["message"]["content"]
            usage = result_json["usage"]
            
            # Update statistics
            self.usage_stats.prompt_tokens += usage["prompt_tokens"]
            self.usage_stats.completion_tokens += usage["completion_tokens"]
            self.usage_stats.image_count += count_images_in_messages(messages)
            
            return result
        else:
            raise Exception(response.json()["error"]["message"])
    
    def call_cua(self, messages: list, **kwargs) -> Tuple[str, str]:
        retry = 3
        for _ in range(retry):
            py_cmd = self.__call__(messages)
            if "```python" in py_cmd:
                py_cmd = py_cmd.split("```python")[1].split("```")[0]
                break
            else:
                print("Invalid response format, retrying: ", py_cmd)
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "You must return a python code within ```python``` code block to execute the task."},
                    ],
                })
        return py_cmd, ""


class AbstractLLM:
    """
    LLM abstraction layer providing unified interface and retry mechanism
    """
    
    def __init__(self, model_name: str, temperature: float = 0.1, max_tokens: int = 4096):
        """
        Initialize LLM instance
        
        Args:
            model_name: Model name
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens
        """
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Model {model_name} not supported")
        
        temperature = max(0.1, temperature)
        self.model_name = model_name
        self.timeout_seconds = 600
        
        # Get model configuration
        self.model_config = MODEL_CONFIGS[model_name]
        
        # Create client instance
        client_class = globals()[self.model_config.client_class]
        self.client = client_class(self.model_config.real_model_name, temperature, max_tokens)
        self.is_vlm = self.model_config.is_vlm
    
    def __call__(self, messages: list, max_retries: int = 3) -> Optional[str]:
        """
        Call LLM with timeout and retry mechanism
        
        Args:
            messages: List of messages
            max_retries: Maximum number of retries
        
        Returns:
            LLM response or None (on failure)
        """
        for attempt in range(max_retries):
            try:
                @with_timeout(self.timeout_seconds)
                def call_llm():
                    return self.client(messages)
                
                response = call_llm()
                return response
            
            except TimeoutError:
                print(f"Attempt {attempt + 1}/{max_retries}: LLM call timed out after {self.timeout_seconds} seconds")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
            
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries}: LLM call failed with error: {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
        
        print(f"All {max_retries} attempts failed")
        return None
    
    def call_cua(self, messages: list, **kwargs) -> Tuple[str, str]:
        """Call Computer Use API"""
        return self.client.call_cua(messages, **kwargs)
    
    def get_usage(self) -> Tuple[float, int, int, int]:
        """
        Get cost and usage statistics
        
        Returns:
            (total_cost, prompt_tokens, completion_tokens, image_count)
        """
        stats = self.client.get_usage_stats()
        cost = self.model_config.calculate_cost(stats)
        return cost, stats.prompt_tokens, stats.completion_tokens, stats.image_count
    
    def reset_stats(self):
        """Reset usage statistics"""
        self.client.reset_stats()


if __name__ == "__main__":
    # Test basic call
    llm = AbstractLLM("qwen3")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
            ],
        }
    ]
    
    response = llm(messages)
    print("Response:", response)
    
    # Get cost information
    cost, prompt_tokens, completion_tokens, image_count = llm.get_usage()
    print(f"\nUsage Statistics:")
    print(f"Cost: ${cost:.6f}")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Completion tokens: {completion_tokens}")
    print(f"Images: {image_count}")
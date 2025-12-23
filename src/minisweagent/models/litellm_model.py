import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import litellm
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.cache_control import set_cache_control

logger = logging.getLogger("litellm_model")

# Default max context length for truncation (in tokens)
# Set to 60k (60*1024) to leave 4k tokens headroom for generation
# when using max_model_len=64k (65536)
DEFAULT_MAX_CONTEXT_LENGTH = 61440


class ContextLengthExceeded(Exception):
    """Raised when conversation exceeds the maximum context length.

    This exception signals that the instance should be discarded or retried,
    rather than truncating the conversation.
    """
    def __init__(self, current_tokens: int, max_tokens: int, message: str = None):
        self.current_tokens = current_tokens
        self.max_tokens = max_tokens
        super().__init__(message or f"Context length exceeded: {current_tokens} tokens > {max_tokens} max")

# Cache for tokenizers to avoid reloading
_TOKENIZER_CACHE = {}

# Build exception tuple based on available exceptions (for LiteLLM version compatibility)
_NO_RETRY_EXCEPTIONS = []
for exc_name in [
    "UnsupportedParamsError",
    "NotFoundError",
    "PermissionDeniedError",
    "ContextWindowExceededError",
    "APIError",
    "AuthenticationError",
]:
    if hasattr(litellm.exceptions, exc_name):
        _NO_RETRY_EXCEPTIONS.append(getattr(litellm.exceptions, exc_name))
_NO_RETRY_EXCEPTIONS.append(KeyboardInterrupt)
_NO_RETRY_EXCEPTIONS.append(ContextLengthExceeded)  # Don't retry context length errors
_NO_RETRY_EXCEPTIONS = tuple(_NO_RETRY_EXCEPTIONS)


def get_tokenizer(model_name: str):
    """
    Get or load tokenizer for a model, with caching.

    Args:
        model_name: Model name (e.g., "openai/gpt-4", "Kwai-Klear/Klear-AgentForge-8B-SFT")

    Returns:
        Tokenizer instance
    """
    if model_name in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_name]

    try:
        from transformers import AutoTokenizer

        # Strip provider prefixes to get actual HuggingFace model path
        actual_model = model_name
        for prefix in ["vllm/", "openai/", "huggingface/"]:
            if actual_model.startswith(prefix):
                actual_model = actual_model[len(prefix):]
                break

        # For API models without a valid HF path, use a fallback
        if "/" not in actual_model or actual_model.startswith("gpt-"):
            actual_model = "gpt2"  # Fallback to GPT-2 tokenizer

        logger.info(f"Loading tokenizer for model: {actual_model}")
        tokenizer = AutoTokenizer.from_pretrained(actual_model, trust_remote_code=True)
        _TOKENIZER_CACHE[model_name] = tokenizer
        return tokenizer
    except Exception as e:
        logger.warning(f"Failed to load tokenizer for {model_name}: {e}. Falling back to character-based estimation.")
        # Return None to signal fallback to character-based counting
        return None


def count_tokens(text: str, tokenizer) -> int:
    """
    Count tokens in text using the tokenizer.

    Args:
        text: Input text
        tokenizer: Tokenizer instance (or None for character-based fallback)

    Returns:
        Token count (or character count / 4 if tokenizer unavailable)
    """
    if tokenizer is None:
        # Fallback: estimate 1 token ≈ 4 characters
        return len(text) // 4

    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception as e:
        logger.warning(f"Tokenization failed: {e}. Using character-based estimation.")
        return len(text) // 4


def check_context_length(
    messages: list[dict[str, str]],
    max_tokens: int = DEFAULT_MAX_CONTEXT_LENGTH,
    tokenizer=None
) -> list[dict[str, str]]:
    """
    Check if messages fit within context length limit.

    Instead of truncating, raises ContextLengthExceeded if the limit is exceeded.
    This allows the caller to decide whether to retry or discard the instance.

    Args:
        messages: List of message dicts with 'role' and 'content'
        max_tokens: Maximum total token count
        tokenizer: Tokenizer instance (optional, will use character-based fallback if None)

    Returns:
        Original messages list if within limit

    Raises:
        ContextLengthExceeded: If messages exceed max_tokens
    """
    if not messages:
        return messages

    # Calculate total token count
    total_tokens = sum(count_tokens(msg.get('content', ''), tokenizer) for msg in messages)

    if total_tokens > max_tokens:
        logger.warning(
            f"Context length exceeded: {total_tokens} tokens > {max_tokens} max tokens. "
            f"Instance will be discarded or retried."
        )
        raise ContextLengthExceeded(total_tokens, max_tokens)

    return messages


# Keep old function name as alias for backwards compatibility
def truncate_messages_to_fit_context(
    messages: list[dict[str, str]],
    max_tokens: int = DEFAULT_MAX_CONTEXT_LENGTH,
    tokenizer=None
) -> list[dict[str, str]]:
    """Alias for check_context_length (no longer truncates, raises exception instead)."""
    return check_context_length(messages, max_tokens, tokenizer)


@dataclass
class LitellmModelConfig:
    model_name: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    litellm_model_registry: Path | str | None = os.getenv("LITELLM_MODEL_REGISTRY_PATH")
    set_cache_control: Literal["default_end"] | None = None
    """Set explicit cache control markers, for example for Anthropic models"""
    max_context_length: int = DEFAULT_MAX_CONTEXT_LENGTH
    """Maximum context length in tokens (default: 60,000 tokens, well below 65,536 limit)"""


class LitellmModel:
    def __init__(self, *, config_class: type = LitellmModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        if self.config.litellm_model_registry and Path(self.config.litellm_model_registry).is_file():
            litellm.utils.register_model(json.loads(Path(self.config.litellm_model_registry).read_text()))

        local_vllm_server_ips_filename = os.getenv("local_vllm_server_ips_filename")
        if local_vllm_server_ips_filename is not None:
            if os.path.exists(fnm):
                self.vllm_urls = [_.strip('\n') for _ in open(fnm).readlines() if len(_)  > 5]
            else:
                self.vllm_urls = []
        else:
            self.vllm_urls = []

        # Load tokenizer for accurate token counting in truncation
        self.tokenizer = get_tokenizer(self.config.model_name)

        # Initialize direct vLLM engine if using vLLM with LoRA
        self.vllm_engine = None
        self.vllm_lora_params = {}
        if self.config.model_name.startswith("vllm/"):
            # Extract LoRA parameters from model_kwargs
            lora_params = {}
            for param in ["enable_lora", "lora_local_path", "max_lora_rank"]:
                if param in self.config.model_kwargs:
                    lora_params[param] = self.config.model_kwargs[param]

            if lora_params.get("enable_lora"):
                logger.info(f"Initializing vLLM engine with LoRA support: {lora_params}")
                self._init_vllm_engine_with_lora(lora_params)
                self.vllm_lora_params = lora_params

    def _init_vllm_engine_with_lora(self, lora_params: dict):
        """Initialize vLLM engine directly with LoRA support"""
        try:
            from vllm import LLM, SamplingParams

            # Extract model path from "vllm/model_path" format
            model_path = self.config.model_name.split("vllm/", 1)[1]

            # Prepare vLLM engine kwargs
            engine_kwargs = {
                "model": model_path,
                "trust_remote_code": self.config.model_kwargs.get("trust_remote_code", True),
                "dtype": self.config.model_kwargs.get("dtype", "bfloat16"),
                "max_model_len": self.config.model_kwargs.get("max_model_len", 65536),
                "tensor_parallel_size": self.config.model_kwargs.get("tensor_parallel_size", 1),
            }

            # Add LoRA parameters
            if lora_params.get("enable_lora"):
                engine_kwargs["enable_lora"] = True
                engine_kwargs["max_lora_rank"] = lora_params.get("max_lora_rank", 16)

            logger.info(f"Creating vLLM LLM engine with kwargs: {engine_kwargs}")
            self.vllm_engine = LLM(**engine_kwargs)

            # Load LoRA adapter if path provided
            if lora_params.get("lora_local_path"):
                logger.info(f"Loading LoRA adapter from: {lora_params['lora_local_path']}")
                # Note: vLLM loads LoRA adapters dynamically per request
                # We just store the path for use in sampling params

            logger.info("vLLM engine with LoRA initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine with LoRA: {e}")
            logger.warning("Falling back to standard LiteLLM integration (LoRA may not work)")
            self.vllm_engine = None

    @retry(
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(_NO_RETRY_EXCEPTIONS),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs):
        # Use direct vLLM engine if initialized with LoRA
        if self.vllm_engine is not None:
            return self._query_vllm_direct(messages, **kwargs)

        # Filter out vLLM engine-init parameters from model_kwargs for LiteLLM
        filtered_model_kwargs = {
            k: v for k, v in self.config.model_kwargs.items()
            if k not in ["enable_lora", "lora_local_path", "max_lora_rank"]
        }

        if len(self.vllm_urls):
            total_server = len(self.vllm_urls)
            server_idx = str_hash_to_int(messages[1]['content']) % total_server
            api_base = self.vllm_urls[server_idx]
        else:
            api_base = None
        try:
            if api_base is not None:
                return litellm.completion(
                    api_base=api_base,
                    model=self.config.model_name, messages=messages, **(filtered_model_kwargs | kwargs)
                )
            else:
                return litellm.completion(
                    model=self.config.model_name, messages=messages, **(filtered_model_kwargs | kwargs)
                )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e
        except litellm.exceptions.APIConnectionError as e:
            # Don't retry if it's a token length error - these won't fix themselves
            if "longer than the maximum model length" in str(e):
                logger.error(f"Token length exceeded max model length. Skipping retries. Error: {e}")
                # Re-raise as ContextWindowExceededError which won't be retried
                raise litellm.exceptions.ContextWindowExceededError(
                    message=str(e),
                    model=self.config.model_name,
                    llm_provider="vllm"
                )
            raise

    def _query_vllm_direct(self, messages: list[dict[str, str]], **kwargs):
        """Query vLLM engine directly (used when LoRA is enabled)"""
        from vllm import SamplingParams

        # Convert messages to prompt string (vLLM expects string prompts)
        # Use tokenizer's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback: simple concatenation
            prompt = "\n\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)

        # Safety check: verify prompt token count before sending to vLLM
        max_model_len = self.config.model_kwargs.get("max_model_len", 65536)
        max_tokens_response = kwargs.get("max_tokens", self.config.model_kwargs.get("max_tokens", 4096))
        prompt_tokens = count_tokens(prompt, self.tokenizer)
        if prompt_tokens + max_tokens_response > max_model_len:
            logger.error(
                f"PROMPT TOO LONG even after truncation! "
                f"Prompt: {prompt_tokens} tokens + max_tokens: {max_tokens_response} = {prompt_tokens + max_tokens_response} > max_model_len: {max_model_len}"
            )
            # Raise a clear error instead of letting vLLM fail with cryptic message
            raise litellm.exceptions.ContextWindowExceededError(
                message=f"Prompt ({prompt_tokens} tokens) + max_tokens ({max_tokens_response}) exceeds max_model_len ({max_model_len})",
                model=self.config.model_name,
                llm_provider="vllm"
            )

        # Prepare sampling parameters
        sampling_kwargs = {
            "temperature": kwargs.get("temperature", self.config.model_kwargs.get("temperature", 0.2)),
            "max_tokens": kwargs.get("max_tokens", self.config.model_kwargs.get("max_tokens", 4096)),
            "top_p": kwargs.get("top_p", self.config.model_kwargs.get("top_p", 1.0)),
        }

        # Add LoRA path if available
        lora_request = None
        if self.vllm_lora_params.get("lora_local_path"):
            try:
                from vllm.lora.request import LoRARequest
                lora_path = self.vllm_lora_params["lora_local_path"]
                lora_request = LoRARequest("default_lora", 1, lora_path)
                logger.debug(f"Using LoRA adapter from: {lora_path}")
            except Exception as e:
                logger.warning(f"Failed to create LoRA request: {e}")

        sampling_params = SamplingParams(**sampling_kwargs)

        # Generate completion
        outputs = self.vllm_engine.generate([prompt], sampling_params, lora_request=lora_request)

        # Convert vLLM output to LiteLLM-style response
        if outputs and len(outputs) > 0:
            output = outputs[0]
            generated_text = output.outputs[0].text if output.outputs else ""

            # Format response to match LiteLLM's structure
            response = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": generated_text
                    },
                    "finish_reason": output.outputs[0].finish_reason if output.outputs else "stop"
                }],
                "model": self.config.model_name,
                "usage": {
                    "prompt_tokens": len(output.prompt_token_ids) if hasattr(output, 'prompt_token_ids') else 0,
                    "completion_tokens": len(output.outputs[0].token_ids) if output.outputs else 0,
                    "total_tokens": 0  # Will be calculated below
                }
            }
            response["usage"]["total_tokens"] = response["usage"]["prompt_tokens"] + response["usage"]["completion_tokens"]

            return response
        else:
            raise ValueError("vLLM engine returned no outputs")

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        # Calculate token count before truncation for logging
        pre_truncation_tokens = sum(count_tokens(msg.get('content', ''), self.tokenizer) for msg in messages)

        # Truncate messages to fit context length before processing
        messages = truncate_messages_to_fit_context(
            messages,
            max_tokens=self.config.max_context_length,
            tokenizer=self.tokenizer
        )

        # Log token count after truncation
        post_truncation_tokens = sum(count_tokens(msg.get('content', ''), self.tokenizer) for msg in messages)
        if pre_truncation_tokens != post_truncation_tokens:
            logger.warning(f"TRUNCATION APPLIED: {pre_truncation_tokens} -> {post_truncation_tokens} tokens (limit: {self.config.max_context_length})")
        else:
            logger.info(f"Token count: {post_truncation_tokens} / {self.config.max_context_length} (no truncation needed)")

        if self.config.set_cache_control:
            messages = set_cache_control(messages, mode=self.config.set_cache_control)
        response = self._query(messages, **kwargs)
        cost = 0
        # try:
        #     cost = litellm.cost_calculator.completion_cost(response)
        # except Exception as e:
        #     logger.critical(
        #         f"Error calculating cost for model {self.config.model_name}: {e}. "
        #         "Please check the 'Updating the model registry' section in the documentation at "
        #         "https://klieret.short.gy/litellm-model-registry Still stuck? Please open a github issue for help!"
        #     )
        #     raise
        self.n_calls += 1
        assert cost >= 0.0, f"Cost is negative: {cost}"
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)

        # Handle both dict (from _query_vllm_direct) and object (from litellm) response formats
        if isinstance(response, dict):
            # Direct vLLM response (dict format)
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            response_dump = response
        else:
            # LiteLLM response (object format)
            content = response.choices[0].message.content or ""  # type: ignore
            response_dump = response.model_dump()

        return {
            "content": content,
            "extra": {
                "response": response_dump,
            },
        }

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}

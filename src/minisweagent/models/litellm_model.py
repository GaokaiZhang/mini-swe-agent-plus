import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import litellm

# Drop unsupported params (e.g., trust_remote_code) instead of raising errors
# This fixes compatibility issues between LiteLLM and vLLM where LiteLLM passes
# params that vLLM's SamplingParams doesn't accept
litellm.drop_params = True

# Handle version differences in litellm exceptions
# UnsupportedParamsError was added in later versions
try:
    UnsupportedParamsError = litellm.exceptions.UnsupportedParamsError
except AttributeError:
    # Create a dummy exception that will never be raised for older litellm versions
    class UnsupportedParamsError(Exception):
        pass

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.cache_control import set_cache_control
from minisweagent.agents.default import ContextWindowExceeded

logger = logging.getLogger("litellm_model")


@dataclass
class LitellmModelConfig:
    model_name: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    litellm_model_registry: Path | str | None = os.getenv("LITELLM_MODEL_REGISTRY_PATH")
    set_cache_control: Literal["default_end"] | None = None
    """Set explicit cache control markers, for example for Anthropic models"""


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

    @retry(
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(
            (
                UnsupportedParamsError,
                litellm.exceptions.NotFoundError,
                litellm.exceptions.PermissionDeniedError,
                litellm.exceptions.ContextWindowExceededError,
                litellm.exceptions.APIError,
                litellm.exceptions.AuthenticationError,
                KeyboardInterrupt,
                ContextWindowExceeded,  # Our custom exception for context exceeded
            )
        ),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs):
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
                    model=self.config.model_name, messages=messages, **(self.config.model_kwargs | kwargs)
                )
            else:
                return litellm.completion(
                    model=self.config.model_name, messages=messages, **(self.config.model_kwargs | kwargs)
                )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e
        except litellm.exceptions.ContextWindowExceededError as e:
            # Convert to our custom exception so it terminates the agent cleanly
            raise ContextWindowExceeded(f"Context window exceeded: {e}") from e
        except litellm.exceptions.InternalServerError as e:
            # vLLM returns 500 errors for context window exceeded, check the message
            error_msg = str(e).lower()
            if "longer than the maximum model length" in error_msg or "context" in error_msg and "exceed" in error_msg:
                raise ContextWindowExceeded(f"Context window exceeded (from 500 error): {e}") from e
            # Re-raise other internal server errors
            raise

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
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
        return {
            "content": response.choices[0].message.content or "",  # type: ignore
            "extra": {
                "response": response.model_dump(),
            },
        }

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}

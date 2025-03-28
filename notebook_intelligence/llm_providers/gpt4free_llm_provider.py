#!notebook_intelligence/llm_providers/gpt4free_llm_provider.py
# -*- coding: utf-8 -*-
"""
Provides an LLM provider implementation for interacting with gpt4free.
"""

from typing import Any
from notebook_intelligence.api import ChatModel, EmbeddingModel, InlineCompletionModel, LLMProvider, CancelToken, ChatResponse, CompletionContext
from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field, validator
try:
    import g4f
    from g4f.errors import ProviderNotFoundError, ModelNotFoundError, RetryProviderError
    from g4f.providers.base_provider import BaseProvider
    from g4f.models import Model, ModelUtils

    _G4F_AVAILABLE = True
except ImportError:
    _G4F_AVAILABLE = False
    # Define dummy types for type hinting if g4f is not installed
    BaseProvider = type("BaseProvider", (), {})
    Model = type("Model", (), {})
    ModelUtils = type("ModelUtils", (), {})
    ProviderNotFoundError = type("ProviderNotFoundError", (Exception,), {})
    ModelNotFoundError = type("ModelNotFoundError", (Exception,), {})
    RetryProviderError = type("RetryProviderError", (Exception,), {})

    logger.warning("gpt4free library not found. Gpt4FreeLLMProvider will not be available.")
    logger.warning("Please install it using: pip install -U gpt4free")

import logging

from notebook_intelligence.util import extract_llm_generated_code

log = logging.getLogger(__name__)

GPT4FREE_EMBEDDING_FAMILIES = set(["claude-3.7-sonnet", "claude-3.7-sonnet"])
GPT4FREE_INLINE_COMPL_PROMPT = """<｜fim▁begin｜>{prefix}<｜fim▁hole｜>{suffix}<｜fim▁end｜>"""

class Gpt4FreeSettings(BaseSettings):
    """
    Configuration settings for the Gpt4FreeLLMProvider.

    Reads settings from environment variables with the prefix 'GPT4FREE_'.
    """

    provider: str = Field("__all__", description="The gpt4free provider name to use (e.g., 'DeepAi', 'You').")
    model: str = Field("claude-3.7-sonnet", description="The model name to use within gpt4free (e.g., 'gpt-3.5-turbo', 'gpt-4').")

    class Config:
        """Pydantic configuration settings."""

        env_prefix = "GPT4FREE_"
        case_sensitive = False


    @validator("provider")
    def validate_provider(cls, v: str) -> str:
        """Validate if the provider string is potentially valid."""
        if not _G4F_AVAILABLE:
            logger.warning("Skipping g4f provider validation as library is not installed.")
            return v
        try:
            # Try resolving the provider class
            _ = getattr(g4f.Provider, v)
        except AttributeError:
            raise ValueError(f"Unknown gpt4free provider name: '{v}'. Available providers might include: {g4f.Provider.__all__}")
        return v

class Gpt4FreeChatModel(ChatModel):
    def __init__(self, provider: LLMProvider, model_id: str, model_name: str, context_window: int):
        super().__init__(provider)
        self._model_id = model_id
        self._model_name = model_name
        self._context_window = context_window

    @property
    def id(self) -> str:
        return self._model_id

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def context_window(self) -> int:
        return self._context_window
    ##TODO
    def completions(self, messages: list[dict], tools: list[dict] = None, response: ChatResponse = None, cancel_token: CancelToken = None, options: dict = {}) -> Any:
        stream = response is not None
        completion_args = {
            "model": self._model_id, 
            "messages": messages.copy(),
            "stream": stream,
        }
        if tools is not None and len(tools) > 0:
            completion_args["tools"] = tools

        return ""
"""
        Gets a completion from the configured gpt4free provider and model.

        Note: `max_tokens` might not be supported by all gpt4free providers.
        Note: `temperature` support varies between gpt4free providers.

        Args:
            prompt: The user's prompt.
            system_message: An optional system message to guide the AI.
            temperature: The sampling temperature.
            max_tokens: The maximum number of tokens to generate (support varies).

        Returns:
            The generated text completion as a string, or None if an error occurred.
"""
"""
def get_completion(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        if not self._provider_instance:
             self.logger.error("Gpt4Free provider instance not resolved during initialization.")
             return None

        messages: List[Dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        self.logger.info(
            f"Requesting completion from g4f provider '{self.settings.provider}' "
            f"model '{self.settings.model}' with temperature {temperature}"
        )
        if max_tokens:
             self.logger.warning("max_tokens parameter is provided but may not be supported by all g4f providers.")
             # Note: g4f.ChatCompletion.create doesn't directly accept max_tokens usually.
             # It might need to be passed via provider-specific options if supported.
             # For now, we log a warning and don't pass it directly.

        try:
            response = g4f.ChatCompletion.create(
                model=self.settings.model,
                messages=messages,
                provider=self._provider_instance,
                temperature=temperature,
                # stream=False, # Default is False
                # Other potential parameters depending on g4f version and provider support
            )
            if isinstance(response, str):
                self.logger.info(f"Received completion from g4f provider '{self.settings.provider}'")
                self.logger.debug(f"Completion result: {response[:100]}...") # Log beginning of response
                return response
            else:
                # Handle cases where the response might be unexpected (e.g., generator if stream=True was used)
                self.logger.error(f"Unexpected response type from g4f: {type(response)}")
                return None

        except (ProviderNotFoundError, ModelNotFoundError) as e:
            self.logger.error(f"g4f configuration error for provider '{self.settings.provider}' / model '{self.settings.model}': {e}")
            return None
        except RetryProviderError as e:
            self.logger.error(f"g4f provider '{self.settings.provider}' failed after multiple retries: {e}")
            return None
        except Exception as e:
            # Catch any other unexpected errors from g4f or network issues
            self.logger.exception(f"An unexpected error occurred while calling gpt4free provider '{self.settings.provider}': {e}", exc_info=True)
            return None
"""


class Gpt4FreeInlineCompletionModel(InlineCompletionModel):
    def __init__(self, provider: LLMProvider, model_id: str, model_name: str, context_window: int, prompt_template: str):
        super().__init__(provider)
        self._model_id = model_id
        self._model_name = model_name
        self._context_window = context_window
        self._prompt_template = prompt_template

    @property
    def id(self) -> str:
        return self._model_id

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def context_window(self) -> int:
        return self._context_window

    def inline_completions(self, prefix, suffix, language, filename, context: CompletionContext, cancel_token: CancelToken) -> str:
        has_suffix = suffix.strip() != ""
        if has_suffix:
            prompt = self._prompt_template.format(prefix=prefix, suffix=suffix.strip())
        else:
            prompt = prefix

        try:
            generate_args = {
                "model": self._model_id, 
                "prompt": prompt,
                "raw": True,
                "options": {
                    'num_predict': 128,
                    "temperature": 0,
                    "stop" : [
                        "<|end▁of▁sentence|>",
                        "<｜end▁of▁sentence｜>",
                        "<|EOT|>",
                        "<EOT>",
                        "\\n",
                        "</s>",
                        "<|eot_id|>",
                    ],
                },
            }
            ## TODO ?
            g4f_response = g4f.generate(**generate_args)
            code = g4f_response.response
            code = extract_llm_generated_code(code)

            return code
        except Exception as e:
            log.error(f"Error occurred while generating using completions ollama: {e}")
            return ""

class Gpt4FreeLLMProvider(LLMProvider):
    def __init__(self):
        super().__init__()
        self._chat_models = []
        self.update_chat_model_list()

    @property
    def id(self) -> str:
        return "gpt4free"

    @property
    def name(self) -> str:
        return "Gpt4Free"

    @property
    def chat_models(self) -> list[ChatModel]:
        return self._chat_models

    @property
    def inline_completion_models(self) -> list[InlineCompletionModel]:
        return [
            Gpt4FreeInlineCompletionModel(self, "claude-3.7-sonnet", "claude-3.7-sonnet", 163840, GPT4FREE_INLINE_COMPL_PROMPT),
        ]

    @property
    def embedding_models(self) -> list[EmbeddingModel]:
        return []

    def get_list(self):
        return [
            Gpt4FreeChatModel("claude-3.7-sonnet", "claude-3.7-sonnet", 163840),
        ]

    def update_chat_model_list(self):
        try:
            ## TODO
            # response = ollama.list()
            self._chat_models.append(
                Gpt4FreeChatModel("claude-3.7-sonnet", "claude-3.7-sonnet", 163840),
            )
        except Exception as e:
            log.error(f"Error updating supported Gpt4Free models: {e}")
"""
            response = self.get_list()
            models = response.models
            self._chat_models = []
            for model in models:
                try:
                    model_family = model.details.family
                    if model_family in GPT4FREE_EMBEDDING_FAMILIES:
                        continue
                    model_show = ollama.show(model.model)
                    model_info = model_show.modelinfo
                    context_window = model_info[f"{model_family}.context_length"]
                    self._chat_models.append(
                        Gpt4FreeChatModel(self, model.model, model.model, context_window)
                    )
                except Exception as e:
                    log.error(f"Error getting Gpt4Free model info {model}: {e}")
"""
        

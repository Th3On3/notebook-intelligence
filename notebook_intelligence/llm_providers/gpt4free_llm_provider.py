#!notebook_intelligence/llm_providers/gpt4free_llm_provider.py
# -*- coding: utf-8 -*-
"""
Provides an LLM provider implementation for interacting with gpt4free.
"""

import logging
from typing import List, Optional, Dict, Any, Type

from pydantic import BaseModel, BaseSettings, Field, validator

from .base_provider import AbstractLLMProvider

# Configure logging
logger = logging.getLogger(__name__)

# Attempt to import g4f and related components
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


class Gpt4FreeSettings(BaseSettings):
    """
    Configuration settings for the Gpt4FreeLLMProvider.

    Reads settings from environment variables with the prefix 'GPT4FREE_'.
    """

    provider: str = Field("DeepAi", description="The gpt4free provider name to use (e.g., 'DeepAi', 'You').")
    model: str = Field("gpt-3.5-turbo", description="The model name to use within gpt4free (e.g., 'gpt-3.5-turbo', 'gpt-4').")

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

    @validator("model")
    def validate_model(cls, v: str) -> str:
        """Basic validation for model string (doesn't guarantee compatibility with chosen provider)."""
        # This validation is limited as g4f's model resolution can be complex.
        # We check if it *might* be a known model string format.
        if not _G4F_AVAILABLE:
             logger.warning("Skipping g4f model validation as library is not installed.")
             return v
        # Simple check, actual compatibility depends on the provider
        if not v:
            raise ValueError("Model name cannot be empty.")
        # You could add more sophisticated checks here if needed,
        # potentially checking against g4f.models if that seems stable.
        return v


class Gpt4FreeLLMProvider(AbstractLLMProvider):
    """
    An LLM provider implementation using the gpt4free library.

    This provider allows using various free LLM services accessible
    through the gpt4free library. Note that the reliability and availability
    of these services can vary.
    """

    def __init__(self, settings: Optional[Gpt4FreeSettings] = None) -> None:
        """
        Initializes the Gpt4FreeLLMProvider.

        Args:
            settings: Configuration settings for the provider. If None, settings
                      are loaded from environment variables.

        Raises:
            ImportError: If the 'gpt4free' library is not installed.
            ValueError: If the configured provider or model is invalid.
        """
        if not _G4F_AVAILABLE:
            raise ImportError(
                "The 'gpt4free' library is required to use Gpt4FreeLLMProvider. "
                "Please install it: pip install -U gpt4free"
            )

        self.settings = settings or Gpt4FreeSettings()
        self.logger = logging.getLogger(__name__)
        self._provider_instance: Optional[Type[BaseProvider]] = None
        self._model_instance: Optional[Model] = None

        self._resolve_provider_and_model()

        self.logger.info(
            f"Initialized Gpt4FreeLLMProvider with provider '{self.settings.provider}' "
            f"and model '{self.settings.model}'"
        )

    def _resolve_provider_and_model(self) -> None:
        """Resolves provider and model instances based on settings."""
        try:
            self._provider_instance = getattr(g4f.Provider, self.settings.provider)
            self.logger.debug(f"Resolved g4f provider: {self._provider_instance}")
        except AttributeError as e:
            self.logger.error(f"Failed to resolve gpt4free provider: {self.settings.provider}")
            raise ValueError(f"Invalid gpt4free provider '{self.settings.provider}' specified in settings.") from e

        try:
            # g4f uses ModelUtils or direct model mappings. We pass the string name.
            # Actual resolution happens inside g4f.ChatCompletion.create
            # Let's store the model *name* string for clarity.
            # We can try resolving using ModelUtils for an early check if desired,
            # but g4f.ChatCompletion.create handles the main logic.
            # For simplicity, we'll rely on the validation in settings and pass the string.
            pass # Model string is stored in self.settings.model
        except Exception as e: # Catch potential errors if we add resolution logic later
             self.logger.error(f"Failed to validate/resolve gpt4free model: {self.settings.model}")
             raise ValueError(f"Invalid gpt4free model '{self.settings.model}' for provider '{self.settings.provider}'.") from e


    def get_completion(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
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

    def get_models(self) -> List[str]:
        """
        Gets a list of available provider names from gpt4free.

        Note: This lists the *providers* available in the g4f library,
              not necessarily models compatible with a specific provider.
              Model availability depends on the chosen provider.

        Returns:
            A list of available gpt4free provider name strings.
        """
        if not _G4F_AVAILABLE:
            self.logger.warning("Cannot list g4f providers because the library is not installed.")
            return []
        try:
            # g4f.Provider.__all__ usually lists the string names of providers
            providers = g4f.Provider.__all__
            if isinstance(providers, list) and all(isinstance(p, str) for p in providers):
                 self.logger.info(f"Found {len(providers)} potential g4f providers.")
                 return providers
            else:
                 self.logger.warning(f"Unexpected format for g4f.Provider.__all__: {type(providers)}. Returning empty list.")
                 return []
        except Exception as e:
            self.logger.exception(f"An error occurred while trying to list gpt4free providers: {e}", exc_info=True)
            return []


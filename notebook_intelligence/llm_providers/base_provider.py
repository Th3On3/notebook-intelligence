#!notebook_intelligence/llm_providers/base_provider.py
# -*- coding: utf-8 -*-
"""
Defines the abstract base class for Large Language Model (LLM) providers.
"""

import abc
from typing import List, Optional


class AbstractLLMProvider(abc.ABC):
    """
    Abstract base class for LLM providers.

    All concrete LLM provider implementations should inherit from this class
    and implement its abstract methods.
    """

    @abc.abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the provider. Specific arguments depend on the implementation.
        """
        pass

    @abc.abstractmethod
    def get_completion(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        """
        Get a completion from the LLM.

        Args:
            prompt: The main input prompt for the LLM.
            system_message: An optional system message to guide the LLM's behavior.
            temperature: Controls the randomness of the output. Lower values make
                         the output more deterministic.
            max_tokens: The maximum number of tokens to generate in the completion.

        Returns:
            The generated text completion as a string, or None if an error occurred
            or the provider could not generate a completion.
        """
        pass

    @abc.abstractmethod
    def get_models(self) -> List[str]:
        """
        Get a list of available model names supported by the provider.

        Returns:
            A list of strings, where each string is an available model name.
        """
        pass


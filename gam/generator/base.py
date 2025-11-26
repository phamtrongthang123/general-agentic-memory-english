from abc import ABC, abstractmethod
from typing import Any

class AbsGenerator(ABC):
    def __init__(
        self,
        config: dict[str, Any],
    ):
        self.config = config

    @abstractmethod
    def generate_single(
        self,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        schema: dict[str, Any] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Generate single response
        Return format: {"text": str, "json": dict|None, "response": dict}
        Note: parameters like temperature, max_tokens are already set in config, no need to pass again
        """
        pass

    @abstractmethod
    def generate_batch(
        self,
        prompts: list[str] | None = None,
        messages_list: list[list[dict[str, str]]] | None = None,
        schema: dict[str, Any] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Generate batch responses
        Return format: [{"text": str, "json": dict|None, "response": dict}, ...]
        Note: parameters like temperature, max_tokens are already set in config, no need to pass again
        """
        pass
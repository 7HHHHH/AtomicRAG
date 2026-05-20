import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from functools import partial
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

from ..utils.config_utils import BaseConfig
from ..utils.llm_utils import TextChatMessage
from ..utils.logging_utils import get_logger



logger = get_logger(__name__)




@dataclass
class LLMConfig:
    _data: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __getattr__(self, key: str) -> Any:
        # Define patterns to ignore for Jupyter/IPython-related attributes
        ignored_prefixes = ("_ipython_", "_repr_")
        if any(key.startswith(prefix) for prefix in ignored_prefixes):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

        if key in self._data:
            return self._data[key]

        logger.error(f"'{self.__class__.__name__}' object has no attribute '{key}'")
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")


    def __setattr__(self, key: str, value: Any) -> None:
        if key == '_data':
            super().__setattr__(key, value)
        else:
            self._data[key] = value

    def __delattr__(self, key: str) -> None:
        if key in self._data:
            del self._data[key]
        else:
            logger.error(f"'{self.__class__.__name__}' object has no attribute '{key}'")
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style key lookup."""
        if key in self._data:
            return self._data[key]
        logger.error(f"'{key}' not found in configuration.")
        raise KeyError(f"'{key}' not found in configuration.")

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dict-style key assignment."""
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        """Allow dict-style key deletion."""
        if key in self._data:
            del self._data[key]
        else:
            logger.error(f"'{key}' not found in configuration.")
            raise KeyError(f"'{key}' not found in configuration.")

    def __contains__(self, key: str) -> bool:
        """Allow usage of 'in' to check for keys."""
        return key in self._data


    def batch_upsert(self, updates: Dict[str, Any]) -> None:
        """Update existing attributes or add new ones from the given dictionary."""
        self._data.update(updates)

    def to_dict(self) -> Dict[str, Any]:
        """Export the configuration as a JSON-serializable dictionary."""
        return self._data

    def to_json(self) -> str:
        """Export the configuration as a JSON string."""
        return json.dumps(self._data)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LLMConfig":
        """Create an LLMConfig instance from a dictionary."""
        instance = cls()
        instance.batch_upsert(config_dict)
        return instance

    @classmethod
    def from_json(cls, json_str: str) -> "LLMConfig":
        """Create an LLMConfig instance from a JSON string."""
        instance = cls()
        instance.batch_upsert(json.loads(json_str))
        return instance

    def __str__(self) -> str:
        """Provide a user-friendly string representation of the configuration."""
        return json.dumps(self._data, indent=4)




class BaseLLM(ABC):
    """Abstract base class for LLMs."""
    global_config: BaseConfig
    llm_name: str # Class name indicating which LLM model to use.
    llm_config: LLMConfig  # Store LLM specific config, init and handled by specifc LLM


    def __init__(self, global_config: Optional[BaseConfig] = None) -> None:
        if global_config is None:
            logger.debug("global config is not given. Using the default ExperimentConfig instance.")
            self.global_config = BaseConfig()
        else: self.global_config = global_config
        logger.debug(f"Loading {self.__class__.__name__} with global_config: {asdict(self.global_config)}")

        self.llm_name = self.global_config.llm_name
        logger.debug(f"Init {self.__class__.__name__}'s llm_name with: {self.llm_name}")


    @abstractmethod
    def _init_llm_config(self) -> None:
        """
        Each LLM model should extract its own running parameters from global_config and raise exception if any mandatory parameter is not defined in global_config.
        This function must init `self.llm_config`.
        """
        pass


    async def ainfer(self, chat: List[TextChatMessage], **kwargs) -> Tuple[str, dict, bool]:
        """
        Perform asynchronous inference using the LLM.

        Base implementation runs the synchronous `infer` method in the default
        executor. Async-native subclasses should override this method.
        """
        loop = asyncio.get_running_loop()
        infer_partial = partial(self.infer, chat, **kwargs)
        return await loop.run_in_executor(None, infer_partial)



    def infer(self, chat: List[TextChatMessage], **kwargs) -> Tuple[str, dict, bool]:
        """
        Perform synchronous inference using the LLM.

        Subclasses must override this method if they provide synchronous
        inference. Async-only subclasses can override `ainfer` and call it
        from here when backwards compatibility is required.
        """
        raise NotImplementedError("Subclasses must implement infer().")


# # Example usage
# if __name__ == "__main__":
#     config = LLMConfig()
#     config.batch_upsert({"learning_rate": 0.001, "batch_size": 32})
#     print(config.to_dict())

#     config.optimizer = "adam"
#     print(config.to_dict())

#     json_config = config.to_json()
#     print(json_config)

#     new_config = LLMConfig.from_json(json_config)
#     print(new_config.to_dict())

#     dict_config = {"dropout": 0.5, "epochs": 10}
#     another_config = LLMConfig.from_dict(dict_config)
#     print(another_config.to_dict())

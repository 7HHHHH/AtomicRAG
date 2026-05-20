import asyncio
import concurrent.futures
import functools
import hashlib
import json
import os
import sqlite3
from copy import deepcopy
from typing import List, Tuple

import httpx
import openai
from filelock import FileLock
from openai import AsyncAzureOpenAI, AsyncOpenAI
from packaging import version
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from ..utils.config_utils import BaseConfig
from ..utils.llm_utils import TextChatMessage
from ..utils.logging_utils import get_logger
from .base import BaseLLM, LLMConfig

logger = get_logger(__name__)


def async_cache_response(func):
    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        use_cache = kwargs.pop("use_cache", True)
        if args:
            messages = args[0]
        else:
            messages = kwargs.get("messages")
        if messages is None:
            raise ValueError("Missing required 'messages' parameter for caching.")

        gen_params = getattr(self, "llm_config", {}).generate_params if hasattr(self, "llm_config") else {}
        model = kwargs.get("model", gen_params.get("model"))
        seed = kwargs.get("seed", gen_params.get("seed"))
        temperature = kwargs.get("temperature", gen_params.get("temperature"))

        key_data = {
            "messages": messages,
            "model": model,
            "seed": seed,
            "temperature": temperature,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()

        if use_cache:
            cache_result = await asyncio.to_thread(self._cache_read_sync, key_hash)
            if cache_result is not None:
                message, metadata = cache_result
                return message, metadata, True

        message, metadata, _ = await func(self, *args, **kwargs)

        if use_cache:
            await asyncio.to_thread(self._cache_write_sync, key_hash, message, metadata)

        return message, metadata, False

    return wrapper


def async_dynamic_retry_decorator(func):
    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        max_retries = getattr(self, "max_retries", 3)
        retryer = AsyncRetrying(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=0.5, min=1, max=10),
        )
        async for attempt in retryer:
            with attempt:
                return await func(self, *args, **kwargs)

    return wrapper


class CacheOpenAI(BaseLLM):
    """OpenAI LLM implementation with async support."""

    @classmethod
    def from_experiment_config(cls, global_config: BaseConfig) -> "CacheOpenAI":
        cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        return cls(
            cache_dir=cache_dir,
            global_config=global_config,
            max_retries=global_config.max_retry_attempts  # Pass retry config to __init__
        )

    def __init__(
        self,
        cache_dir,
        global_config,
        cache_filename: str = None,
        high_throughput: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(global_config=global_config)
        self.cache_dir = cache_dir
        self.global_config = global_config

        self.llm_name = global_config.llm_name
        self.llm_base_url = global_config.llm_base_url

        os.makedirs(self.cache_dir, exist_ok=True)
        if cache_filename is None:
            cache_filename = f"{self.llm_name.replace('/', '_')}_cache.sqlite"
        self.cache_file_name = os.path.join(self.cache_dir, cache_filename)
        self._cache_lock_file = self.cache_file_name + ".lock"
        self._ensure_cache_table()

        self._init_llm_config()

        self._http_client = None
        if high_throughput:
            max_conn = max(global_config.max_concurrency, 500)
            keepalive_conn = max(int(max_conn * 0.2), 100)
            limits = httpx.Limits(
                max_connections=max_conn,
                max_keepalive_connections=keepalive_conn,
            )
            logger.info(
                f"HTTP connection pool: max_connections={max_conn}, keepalive={keepalive_conn}"
            )
            self._http_client = httpx.AsyncClient(
                limits=limits,
                timeout=httpx.Timeout(60.0, read=60.0, connect=10.0),
            )

        self.max_retries = kwargs.get("max_retries", 3)

        client_kwargs = {
            "max_retries": self.max_retries,
            "timeout": httpx.Timeout(60.0, read=60.0, connect=10.0),
        }
        if self._http_client is not None:
            client_kwargs["http_client"] = self._http_client

        if self.global_config.azure_endpoint is None:
            self.openai_client = AsyncOpenAI(
                base_url=self.llm_base_url,
                **client_kwargs,
            )
        else:
            client_kwargs.update(
                {
                    "api_version": self.global_config.azure_endpoint.split("api-version=")[1],
                    "azure_endpoint": self.global_config.azure_endpoint,
                }
            )
            self.openai_client = AsyncAzureOpenAI(**client_kwargs)

    def _ensure_cache_table(self) -> None:
        try:
            conn = sqlite3.connect(self.cache_file_name, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
                """
            )
            conn.commit()
        except sqlite3.Error as exc:
            logger.warning(f"Failed to initialize cache DB {self.cache_file_name}: {exc}")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _cache_read_sync(self, key_hash: str):
        try:
            conn = sqlite3.connect(self.cache_file_name, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            cursor = conn.execute(
                "SELECT message, metadata FROM cache WHERE key = ?", (key_hash,)
            )
            row = cursor.fetchone()
            conn.close()
            if row is not None:
                message, metadata_str = row
                metadata = json.loads(metadata_str)
                return message, metadata
        except sqlite3.Error as exc:
            logger.debug(f"Cache read error: {exc}")
        return None

    def _cache_write_sync(self, key_hash: str, message: str, metadata: dict):
        """
        Write to cache with improved lock handling for high concurrency.

        Args:
            key_hash: Cache key
            message: Response message
            metadata: Response metadata
        """
        try:
            # Increased timeout from 10s to 60s to handle high concurrency (max_concurrency=2000)
            # In high concurrency scenarios with query decomposition, many coroutines
            # may attempt to write cache simultaneously, requiring longer lock wait times
            with FileLock(self._cache_lock_file, timeout=60):
                conn = sqlite3.connect(self.cache_file_name, timeout=30.0)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                metadata_str = json.dumps(metadata)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cache (key, message, metadata)
                    VALUES (?, ?, ?)
                    """,
                    (key_hash, message, metadata_str),
                )
                conn.commit()
                conn.close()
        except TimeoutError:
            # Lock acquisition timeout - log but don't fail the request
            # The response is still valid even if caching fails
            logger.debug(f"Cache write timeout (lock busy) - response still valid")
        except Exception as exc:
            # Other errors - log but don't fail the request
            logger.debug(f"Failed to cache result: {exc}")

    async def aclose(self):
        if self._http_client is not None:
            await self._http_client.aclose()

    def _init_llm_config(self) -> None:
        config_dict = self.global_config.__dict__

        config_dict["llm_name"] = self.global_config.llm_name
        config_dict["llm_base_url"] = self.global_config.llm_base_url
        config_dict["generate_params"] = {
            "model": self.global_config.llm_name,
            "max_completion_tokens": config_dict.get("max_new_tokens", 400),
            "n": config_dict.get("num_gen_choices", 1),
            "seed": config_dict.get("seed", 0),
            "temperature": config_dict.get("temperature", 0.0),
        }

        self.llm_config = LLMConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s llm_config: {self.llm_config}")

    @async_cache_response
    @async_dynamic_retry_decorator
    async def ainfer(
        self,
        messages: List[TextChatMessage],
        use_cache: bool = True,
        **kwargs,
    ) -> Tuple[str, dict, bool]:
        params = deepcopy(self.llm_config.generate_params)
        if kwargs:
            params.update(kwargs)
        params["messages"] = messages
        logger.debug(f"Calling OpenAI GPT API with:\n{params}")

        if "gpt" not in params["model"] or version.parse(openai.__version__) < version.parse(
            "1.45.0"
        ):
            params["max_tokens"] = params.pop("max_completion_tokens")

        response = await self.openai_client.chat.completions.create(**params)

        response_message = response.choices[0].message.content
        assert isinstance(response_message, str), "response_message should be a string"

        metadata = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "finish_reason": response.choices[0].finish_reason,
        }

        return response_message, metadata, False

    def infer(self, messages: List[TextChatMessage], **kwargs) -> Tuple[str, dict, bool]:
        coro = self.ainfer(messages=messages, **kwargs)
        try:
            asyncio.get_running_loop()  # Check if there's a running loop
        except RuntimeError:
            return asyncio.run(coro)

        def _run():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
                asyncio.set_event_loop(None)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(_run).result()

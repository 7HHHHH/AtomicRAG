from copy import deepcopy
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig

logger = get_logger(__name__)

def mean_pooling(token_embeddings, mask):
    """Mean pooling for BGE models"""
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

class BGEEmbeddingModel(BaseEmbeddingModel):
    """BGE (BAAI General Embedding) model implementation"""

    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
            logger.debug(
                f"Overriding {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}")

        self._init_embedding_config()

        # Initializing the embedding model
        logger.debug(
            f"Initializing {self.__class__.__name__}'s embedding model with params: {self.embedding_config.model_init_params}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(**self.embedding_config.model_init_params)
        self.embedding_model.eval()
        self.embedding_dim = self.embedding_model.config.hidden_size

    def _init_embedding_config(self) -> None:
        """
        Extract embedding model-specific parameters to init the EmbeddingConfig.

        Returns:
            None
        """

        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            "model_init_params": {
                "pretrained_model_name_or_path": self.embedding_model_name,
                "trust_remote_code": True,
                "torch_dtype": self.global_config.embedding_model_dtype,
                'device_map': "auto",  # added this line to use multiple GPUs
            },
            "encode_params": {
                "max_length": self.global_config.embedding_max_seq_len,
                "instruction": "",
                "batch_size": self.global_config.embedding_batch_size,
                "num_workers": 32
            },
        }

        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s embedding_config: {self.embedding_config}")

    def encode(self, texts: List[str], instruction: str = "", **kwargs) -> np.ndarray:
        """
        Encode texts using BGE model with mean pooling.

        Args:
            texts: List of texts to encode
            instruction: Instruction for BGE models (optional)
            **kwargs: Additional parameters

        Returns:
            numpy.ndarray: Encoded embeddings
        """
        if instruction:
            texts = [f"{instruction}{text}" for text in texts]

        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.embedding_config.encode_params["max_length"],
            return_tensors='pt'
        )

        # Move input tensors to the same device as the model
        device = next(self.embedding_model.parameters()).device
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

        with torch.no_grad():
            model_output = self.embedding_model(**encoded_input)
            sentence_embeddings = mean_pooling(model_output.last_hidden_state, encoded_input['attention_mask'])

        if self.embedding_config.norm:
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings.cpu().numpy()

    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        Batch encode texts using BGE model with batching support.

        Args:
            texts: List of texts to encode
            **kwargs: Additional parameters (instruction, batch_size, etc.)

        Returns:
            numpy.ndarray: Encoded embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        params = deepcopy(self.embedding_config.encode_params)
        if kwargs:
            params.update(kwargs)

        instruction = params.pop("instruction", "")
        batch_size = params.pop("batch_size", 16)

        logger.debug(f"Calling {self.__class__.__name__} batch_encode with batch_size={batch_size}")

        if len(texts) <= batch_size:
            return self.encode(texts, instruction=instruction, **params)
        else:
            pbar = tqdm(total=len(texts), desc="Batch Encoding")
            results = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.encode(batch_texts, instruction=instruction, **params)
                results.append(batch_embeddings)
                pbar.update(len(batch_texts))
            pbar.close()
            return np.vstack(results)

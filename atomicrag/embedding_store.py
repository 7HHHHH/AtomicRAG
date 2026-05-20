import numpy as np
from tqdm import tqdm
import os
from typing import List
import logging
from copy import deepcopy
import pandas as pd

from .utils.misc_utils import compute_mdhash_id

logger = logging.getLogger(__name__)

class EmbeddingStore:
    def __init__(self, embedding_model, db_filename, batch_size, namespace):
        """
        Initializes the class with necessary configurations and sets up the working directory.

        Parameters:
        embedding_model: The model used for embeddings.
        db_filename: The directory path where data will be stored or retrieved.
        batch_size: The batch size used for processing.
        namespace: A unique identifier for data segregation.

        Functionality:
        - Assigns the provided parameters to instance variables.
        - Checks if the directory specified by `db_filename` exists.
          - If not, creates the directory and logs the operation.
        - Constructs the filename for storing data in a parquet file format.
        - Calls the method `_load_data()` to initialize the data loading process.
        """
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.namespace = namespace

        if not os.path.exists(db_filename):
            logger.info(f"Creating working directory: {db_filename}")
            os.makedirs(db_filename, exist_ok=True)

        self.filename = os.path.join(
            db_filename, f"vdb_{self.namespace}.parquet"
        )
        self._load_data()

    def insert_strings(self, texts: List[str]):
        # 创建一个字典用于存储文本及其对应的哈希ID
        nodes_dict = {}

        # 遍历输入的文本列表，为每个文本生成哈希ID，并存入字典
        for text in texts:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}

        # 获取所有生成的哈希ID
        all_hash_ids = list(nodes_dict.keys())
        # 如果没有哈希ID，说明没有文本需要插入，直接返回
        if not all_hash_ids:
            return  # Nothing to insert.

        # 获取当前已存在的哈希ID集合
        existing = self.hash_id_to_row.keys()

        # 找出还未存在于存储中的哈希ID（即需要插入的新文本）
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]

        # 记录日志，说明有多少新记录需要插入，多少已存在
        logger.info(
            f"Inserting {len(missing_ids)} new records, {len(all_hash_ids) - len(missing_ids)} records already exist.")

        # 如果没有需要插入的新记录，则返回空字典
        if not missing_ids:
            return  {} # All records already exist.

        # 根据缺失的哈希ID，准备需要编码的文本内容
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]

        # 使用嵌入模型批量编码这些文本，得到嵌入向量
        missing_embeddings = self.embedding_model.batch_encode(texts_to_encode)

        # 将新的哈希ID、文本和嵌入向量插入存储
        self._upsert(missing_ids, texts_to_encode, missing_embeddings)

    def _load_data(self):
        if os.path.exists(self.filename):
            df = pd.read_parquet(self.filename)
            self.hash_ids, self.texts, self.embeddings = df["hash_id"].values.tolist(), df["content"].values.tolist(), df["embedding"].values.tolist()
            self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
            self.hash_id_to_row = {
                h: {"hash_id": h, "content": t}
                for h, t in zip(self.hash_ids, self.texts)
            }
            self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
            self.text_to_hash_id = {self.texts[idx]: h  for idx, h in enumerate(self.hash_ids)}
            assert len(self.hash_ids) == len(self.texts) == len(self.embeddings)
            logger.info(f"Loaded {len(self.hash_ids)} records from {self.filename}")
        else:
            self.hash_ids, self.texts, self.embeddings = [], [], []
            self.hash_id_to_idx, self.hash_id_to_row = {}, {}

    def _save_data(self):
        data_to_save = pd.DataFrame({
            "hash_id": self.hash_ids,
            "content": self.texts,
            "embedding": self.embeddings
        })
        data_to_save.to_parquet(self.filename, index=False)
        self.hash_id_to_row = {h: {"hash_id": h, "content": t} for h, t, e in zip(self.hash_ids, self.texts, self.embeddings)}
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
        self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
        logger.info(f"Saved {len(self.hash_ids)} records to {self.filename}")

    def _upsert(self, hash_ids, texts, embeddings):
        self.embeddings.extend(embeddings)
        self.hash_ids.extend(hash_ids)
        self.texts.extend(texts)

        logger.info(f"Saving new records.")
        self._save_data()

    def get_row(self, hash_id):
        return self.hash_id_to_row[hash_id]

    def get_all_ids(self):
        return deepcopy(self.hash_ids)

    def get_all_id_to_rows(self):
        return deepcopy(self.hash_id_to_row)

    def get_embeddings(self, hash_ids, dtype=np.float32) -> list[np.ndarray]:
        if not hash_ids:
            return []

        indices = np.array([self.hash_id_to_idx[h] for h in hash_ids], dtype=np.intp)
        embeddings = np.array(self.embeddings, dtype=dtype)[indices]

        return embeddings

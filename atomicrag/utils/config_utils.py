import os
from dataclasses import dataclass, field
from typing import (
    Literal,
    Union,
    Optional
)

from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class BaseConfig:
    """One and only configuration."""
    # LLM specific attributes
    llm_name: str = field(
        default="gpt-4o-mini",
        metadata={"help": "Class name indicating which LLM model to use."}
    )
    llm_base_url: str = field(
        default=None,
        metadata={"help": "Base URL for the LLM model, if none, means using OPENAI service."}
    )
    embedding_base_url: str = field(
        default=None,
        metadata={"help": "Base URL for an OpenAI compatible embedding model, if none, means using OPENAI service."}
    )
    azure_endpoint: str = field(
        default=None,
        metadata={"help": "Azure Endpoint URI for the LLM model, if none, uses OPENAI service directly."}
    )
    azure_embedding_endpoint: str = field(
        default=None,
        metadata={"help": "Azure Endpoint URI for the OpenAI embedding model, if none, uses OPENAI service directly."}
    )
    max_new_tokens: Union[None, int] = field(
        default=2048,
        metadata={"help": "Max new tokens to generate in each inference."}
    )
    num_gen_choices: int = field(
        default=1,
        metadata={"help": "How many chat completion choices to generate for each input message."}
    )
    seed: Union[None, int] = field(
        default=None,
        metadata={"help": "Random seed."}
    )
    temperature: float = field(
        default=0,
        metadata={"help": "Temperature for sampling in each inference."}
    )

    ## LLM specific attributes -> Async hyperparameters
    max_retry_attempts: int = field(
        default=3,
        metadata={"help": "Max number of retry attempts for an asynchronous API calling."}
    )
    max_concurrency: int = field(
        default=200,
        metadata={"help": "Global maximum number of concurrent workers used across pipeline stages."}
    )
    retrieval_max_workers: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum workers for retrieval thread pool. None keeps default serial mode."}
    )
    # Storage specific attributes
    force_openie_from_scratch: bool = field(
        default=False,
        metadata={"help": "If set to True, will ignore all existing openie files and rebuild them from scratch."}
    )

    # Storage specific attributes
    force_index_from_scratch: bool = field(
        default=False,
        metadata={"help": "If set to True, will ignore all existing storage files and graph data and will rebuild from scratch."}
    )
    passage_node_weight: float = field(
        default=0.1,
        metadata={"help": "Multiplicative factor that modified the passage node weights in PPR."}
    )
    save_openie: bool = field(
        default=True,
        metadata={"help": "If set to True, will save the OpenIE model to disk."}
    )

    # Information extraction specific attributes
    information_extraction_model_name: Literal["openie_openai_gpt", ] = field(
        default="openie_openai_gpt",
        metadata={"help": "Class name indicating which information extraction model to use."}
    )
    skip_graph: bool = field(
        default=False,
        metadata={"help": "Whether to skip graph construction or not (useful for quick retries when embeddings are already available)."}
    )
    unified_extraction: bool = field(
        default=True,
        metadata={"help": "Whether to use unified triple and knowledge fragment extraction in a single LLM call. When True, reduces API calls from 3 to 2 per chunk."}
    )


    # Embedding specific attributes
    embedding_model_name: str = field(
        default="BAAI/bge-large-en-v1.5",
        metadata={"help": "Class name indicating which embedding model to use."}
    )
    embedding_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size of calling embedding model."}
    )
    embedding_return_as_normalized: bool = field(
        default=True,
        metadata={"help": "Whether to normalize encoded embeddings not."}
    )
    embedding_max_seq_len: int = field(
        default=2048,
        metadata={"help": "Max sequence length for the embedding model."}
    )
    embedding_model_dtype: Literal["float16", "float32", "bfloat16", "auto"] = field(
        default="auto",
        metadata={"help": "Data type for local embedding model."}
    )



    # Graph construction specific attributes
    synonymy_edge_topk: int = field(
        default=2047,
        metadata={"help": "k for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_query_batch_size: int = field(
        default=1000,
        metadata={"help": "Batch size for query embeddings for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_key_batch_size: int = field(
        default=10000,
        metadata={"help": "Batch size for key embeddings for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_sim_threshold: float = field(
        default=0.8,
        metadata={"help": "Similarity threshold to include candidate synonymy nodes."}
    )
    is_directed_graph: bool = field(
        default=False,
        metadata={"help": "Whether the graph is directed or not."}
    )

    # Entity seeding for PPR
    entity_node_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for entity seeds when mixing into PPR reset vector."}
    )
    entity_top_k: int = field(
        default=20,
        metadata={"help": "Max number of entity nodes to keep per query when seeding PPR."}
    )
    entity_sim_threshold: float = field(
        default=0.3,
        metadata={"help": "Minimum normalized similarity for an entity to be seeded into PPR."}
    )
    retrieval_top_k: int = field(
        default=25,
        metadata={"help": "Retrieving k documents at each step"}
    )
    damping: float = field(
        default=0.3,
        metadata={"help": "Damping factor for ppr algorithm."}
    )

    # QA specific attributes
    qa_top_k: int = field(
        default=25,
        metadata={"help": "Feeding top k documents to the QA model for reading."}
    )
    qa_prompt_template: Optional[str] = field(
        default=None,
        metadata={"help": "Override the QA prompt template name (without the `rag_qa_` prefix)."}
    )

    # Feature toggles
    enable_fragment_filter: bool = field(
        default=True,
        metadata={"help": "Enable LLM-based fragment filtering before QA generation to remove irrelevant knowledge fragments"}
    )
    enable_query_decomposition: bool = field(
        default=True,
        metadata={"help": "Toggle LLM-based query decomposition for complex questions"}
    )
    enable_ppr: bool = field(
        default=True,
        metadata={"help": "Enable Personalized PageRank graph reasoning; when False, fall back to DPR-only retrieval"}
    )

    # Save dir (highest level directory)
    save_dir: str = field(
        default=None,
        metadata={"help": "Directory to save all related information. If it's given, will overwrite all default save_dir setups. If it's not given, then if we're not running specific datasets, default to `outputs`, otherwise, default to a dataset-customized output dir."}
    )



    # Dataset running specific attributes
    ## Dataset running specific attributes -> General
    dataset: Optional[Literal['hotpotqa', 'hotpotqa_train', 'musique', '2wikimultihopqa']] = field(
        default=None,
        metadata={"help": "Dataset to use. If specified, it means we will run specific datasets. If not specified, it means we're running freely."}
    )
    corpus_len: Optional[int] = field(
        default=None,
        metadata={"help": "Length of the corpus to use."}
    )


    def __post_init__(self):
        if self.save_dir is None: # If save_dir not given
            if self.dataset is None: self.save_dir = 'outputs' # running freely
            else: self.save_dir = os.path.join('outputs', self.dataset) # customize your dataset's output dir here
        logger.debug(f"Initializing the highest level of save_dir to be {self.save_dir}")

import json
import os
import logging
import asyncio
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Set, Dict, Any, Tuple
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from igraph import Graph
import igraph as ig
import re
import time

from .llm import _get_llm_class, BaseLLM
from .embedding_model import _get_embedding_model_class, BaseEmbeddingModel
from .embedding_store import EmbeddingStore
from .information_extraction import OpenIE
from .prompts.linking import get_query_instruction
from .prompts.prompt_template_manager import PromptTemplateManager
from .fragment_filter import FragmentFilter, FragmentFilterRequest
from .utils.misc_utils import *
from .utils.misc_utils import NerRawOutput, TripleRawOutput, KnowledgeFragmentRawOutput
from .utils.embed_utils import retrieve_knn
from .utils.typing import Triple
from .utils.config_utils import BaseConfig
from .query_decomposition import QueryDecomposer, DecompositionResult

logger = logging.getLogger(__name__)


@dataclass
class UsageStatistics:
    """Statistics collector for AtomicRAG operations."""

    # Basic info
    corpus_name: str = ""
    num_questions: int = 0
    num_documents: int = 0
    start_time: str = ""
    end_time: str = ""

    # Time statistics (in seconds)
    total_time: float = 0.0
    indexing_time: float = 0.0
    retrieval_time: float = 0.0
    ppr_time: float = 0.0
    qa_time: float = 0.0
    query_decomposition_time: float = 0.0
    fragment_filter_time: float = 0.0

    # Token statistics
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0

    # Token breakdown by stage
    openie_prompt_tokens: int = 0
    openie_completion_tokens: int = 0
    decomposition_prompt_tokens: int = 0
    decomposition_completion_tokens: int = 0
    filter_prompt_tokens: int = 0
    filter_completion_tokens: int = 0
    qa_prompt_tokens: int = 0
    qa_completion_tokens: int = 0

    # Performance metrics
    avg_retrieval_time_per_query: float = 0.0
    avg_tokens_per_query: float = 0.0
    queries_decomposed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            "basic_info": {
                "corpus_name": self.corpus_name,
                "num_questions": self.num_questions,
                "num_documents": self.num_documents,
                "start_time": self.start_time,
                "end_time": self.end_time,
            },
            "time_statistics": {
                "total_time": round(self.total_time, 2),
                "indexing_time": round(self.indexing_time, 2),
                "retrieval_time": round(self.retrieval_time, 2),
                "ppr_time": round(self.ppr_time, 2),
                "qa_time": round(self.qa_time, 2),
                "query_decomposition_time": round(self.query_decomposition_time, 2),
                "fragment_filter_time": round(self.fragment_filter_time, 2),
            },
            "token_statistics": {
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_tokens,
                "breakdown": {
                    "openie": {
                        "prompt_tokens": self.openie_prompt_tokens,
                        "completion_tokens": self.openie_completion_tokens,
                    },
                    "decomposition": {
                        "prompt_tokens": self.decomposition_prompt_tokens,
                        "completion_tokens": self.decomposition_completion_tokens,
                    },
                    "filter": {
                        "prompt_tokens": self.filter_prompt_tokens,
                        "completion_tokens": self.filter_completion_tokens,
                    },
                    "qa": {
                        "prompt_tokens": self.qa_prompt_tokens,
                        "completion_tokens": self.qa_completion_tokens,
                    },
                },
            },
            "performance_metrics": {
                "avg_retrieval_time_per_query": round(self.avg_retrieval_time_per_query, 4),
                "avg_tokens_per_query": round(self.avg_tokens_per_query, 2),
                "queries_decomposed": self.queries_decomposed,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
            },
        }

    def calculate_totals(self):
        """Calculate total tokens and averages."""
        self.total_prompt_tokens = (
            self.openie_prompt_tokens +
            self.decomposition_prompt_tokens +
            self.filter_prompt_tokens +
            self.qa_prompt_tokens
        )
        self.total_completion_tokens = (
            self.openie_completion_tokens +
            self.decomposition_completion_tokens +
            self.filter_completion_tokens +
            self.qa_completion_tokens
        )
        self.total_tokens = self.total_prompt_tokens + self.total_completion_tokens

        if self.num_questions > 0:
            self.avg_retrieval_time_per_query = self.retrieval_time / self.num_questions
            self.avg_tokens_per_query = self.total_tokens / self.num_questions

class AtomicRAG:

    def __init__(self,
                 global_config=None,
                 save_dir=None,
                 llm_model_name=None,
                 llm_base_url=None,
                 embedding_model_name=None,
                 embedding_base_url=None,
                 azure_endpoint=None,
                 azure_embedding_endpoint=None):
        """
        Initializes an instance of the class and its related components.

        Attributes:
            global_config (BaseConfig): The global configuration settings for the instance. An instance
                of BaseConfig is used if no value is provided.
            saving_dir (str): The directory where specific AtomicRAG instances will be stored. This defaults
                to `outputs` if no value is provided.
            llm_model (BaseLLM): The language model used for processing based on the global
                configuration settings.
            openie (OpenIE): The Open Information Extraction module configured according to the global settings.
            graph: The graph instance initialized by the `initialize_graph` method.
            embedding_model (BaseEmbeddingModel): The embedding model associated with the current
                configuration.
            chunk_embedding_store (EmbeddingStore): The embedding store handling chunk embeddings.
            entity_embedding_store (EmbeddingStore): The embedding store handling entity embeddings.
            hierarchical_fragment_store (EmbeddingStore): The embedding store handling fragment embeddings.
            prompt_template_manager (PromptTemplateManager): The manager for handling prompt templates
                and roles mappings.
            openie_results_path (str): The file path for storing Open Information Extraction results
                based on the dataset and LLM name in the global configuration.
            ready_to_retrieve (bool): A flag indicating whether the system is ready for retrieval
                operations.

        Parameters:
            global_config: The global configuration object. Defaults to None, leading to initialization
                of a new BaseConfig object.
            working_dir: The directory for storing working files. Defaults to None, constructing a default
                directory based on the class name and timestamp.
            llm_model_name: LLM model name, can be inserted directly as well as through configuration file.
            embedding_model_name: Embedding model name, can be inserted directly as well as through configuration file.
            llm_base_url: LLM URL for a deployed LLM model, can be inserted directly as well as through configuration file.
        """
        if global_config is None:
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config

        #Overwriting Configuration if Specified
        if save_dir is not None:
            self.global_config.save_dir = save_dir

        if llm_model_name is not None:
            self.global_config.llm_name = llm_model_name

        if embedding_model_name is not None:
            self.global_config.embedding_model_name = embedding_model_name

        if llm_base_url is not None:
            self.global_config.llm_base_url = llm_base_url
            self.global_config.embedding_base_url = llm_base_url

        if embedding_base_url is not None:
            self.global_config.embedding_base_url = embedding_base_url

        if azure_endpoint is not None:
            self.global_config.azure_endpoint = azure_endpoint

        if azure_embedding_endpoint is not None:
            self.global_config.azure_embedding_endpoint = azure_embedding_endpoint

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self.global_config).items()])
        logger.debug(f"AtomicRAG init with config:\n  {_print_config}\n")

        #LLM and embedding model specific working directories are created under every specified saving directories
        llm_label = self.global_config.llm_name.replace("/", "_")
        embedding_label = self.global_config.embedding_model_name.replace("/", "_")
        self.working_dir = os.path.join(self.global_config.save_dir, f"{llm_label}_{embedding_label}")

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory: {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)

        self.llm_model: BaseLLM = _get_llm_class(self.global_config)

        # Ensure OpenIE shares the same global_config so its semaphores respect max_concurrency
        self.openie = OpenIE(llm_model=self.llm_model, global_config=self.global_config)

        self.graph = self.initialize_graph()

        self.embedding_model: BaseEmbeddingModel = _get_embedding_model_class(
            embedding_model_name=self.global_config.embedding_model_name)(global_config=self.global_config,
                                                                          embedding_model_name=self.global_config.embedding_model_name)
        # Fragment-based architecture: fragments are the primary retrieval unit
        self.entity_embedding_store = EmbeddingStore(self.embedding_model,
                                                     os.path.join(self.working_dir, "entity_embeddings"),
                                                     self.global_config.embedding_batch_size, 'entity')
        # fact_embedding_store removed - fact embeddings are not used in retrieval
        self.hierarchical_fragment_store = EmbeddingStore(
            self.embedding_model,
            os.path.join(self.working_dir, "fragment_embeddings"),
            self.global_config.embedding_batch_size,
            'fragment'
        )

        # Mapping: entity -> fragments (replaces ent_node_to_chunk_ids)
        self.ent_node_to_fragment_ids = {}

        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})

        self.openie_results_path = os.path.join(self.global_config.save_dir,f'openie_results_ner_{self.global_config.llm_name.replace("/", "_")}.json')

        # Initialize Fragment Filter for LLM-based fragment filtering
        self.fragment_filter = FragmentFilter(self)

        self.ready_to_retrieve = False

        # Simple caches to avoid repeat NER/embedding for identical queries
        self._query_entity_cache = {}
        self._query_entity_score_cache = {}
        self._query_entity_emb_cache = {}
        self._entity_text_emb_cache = {}

        self.ppr_time = 0
        self.all_retrieval_time = 0

        # Store mapping from original doc to fragments for traceability
        self.doc_to_fragment_ids = {}

        # Thread safety lock for igraph operations
        # python-igraph is NOT thread-safe, so we need to protect concurrent access to self.graph
        self._graph_lock = threading.RLock()
        self._metrics_lock = threading.Lock()

        # Initialize Query Decomposer for complex question handling
        # Query Decomposition: Key innovation for handling complex multi-hop questions
        # Optimized parameters to balance decomposition benefits with performance:
        # - max_sub_questions: 3 (reduced from 5 to avoid document dilution)
        # - complexity_threshold: 6.5 (increased from 5.0 to only decompose truly complex questions)
        # - Timeout handling: 30s API timeout with 5 retries (max 150s per query)
        self.query_decomposer = QueryDecomposer(
            llm_model=self.llm_model,
            max_sub_questions=3,
            complexity_threshold=6.5,
            prompt_template_manager=self.prompt_template_manager
        )
        logger.info(
            "Initialized QueryDecomposer with optimized parameters (threshold=6.5, max_subs=3)"
        )

        # Initialize statistics collector
        self.stats = UsageStatistics()
        self._start_timestamp = time.time()  # Save raw timestamp for total_time calculation
        self.stats.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _run_async(self, coro):
        """
        Helper to run an async coroutine from sync contexts.
        Ensures compatibility when no loop exists or when already inside one.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        def _runner():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
                asyncio.set_event_loop(None)

        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(_runner).result()

    def _update_token_stats(self, metadata_list: List[Dict], stage: str):
        """
        Update token statistics from LLM metadata.

        Args:
            metadata_list: List of metadata dicts from LLM responses
            stage: Stage name ('openie', 'decomposition', 'filter', 'qa')
        """
        prompt_tokens = sum(m.get('prompt_tokens', 0) for m in metadata_list if isinstance(m, dict))
        completion_tokens = sum(m.get('completion_tokens', 0) for m in metadata_list if isinstance(m, dict))

        if stage == 'openie':
            self.stats.openie_prompt_tokens += prompt_tokens
            self.stats.openie_completion_tokens += completion_tokens
        elif stage == 'decomposition':
            self.stats.decomposition_prompt_tokens += prompt_tokens
            self.stats.decomposition_completion_tokens += completion_tokens
        elif stage == 'filter':
            self.stats.filter_prompt_tokens += prompt_tokens
            self.stats.filter_completion_tokens += completion_tokens
        elif stage == 'qa':
            self.stats.qa_prompt_tokens += prompt_tokens
            self.stats.qa_completion_tokens += completion_tokens

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current usage statistics.

        Returns:
            Dictionary containing all statistics
        """
        self.stats.end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.stats.total_time = time.time() - self._start_timestamp  # Calculate total elapsed time
        self.stats.ppr_time = self.ppr_time
        self.stats.retrieval_time = self.all_retrieval_time
        self.stats.calculate_totals()
        return self.stats.to_dict()


    def initialize_graph(self):
        """
        Initializes a graph using a Pickle file if available or creates a new graph.

        The function attempts to load a pre-existing graph stored in a Pickle file. If the file
        is not present or the graph needs to be created from scratch, it initializes a new directed
        or undirected graph based on the global configuration. If the graph is loaded successfully
        from the file, pertinent information about the graph (number of nodes and edges) is logged.

        Returns:
            ig.Graph: A pre-loaded or newly initialized graph.

        Raises:
            None
        """
        self._graph_pickle_filename = os.path.join(self.working_dir, f"graph.pickle")

        preloaded_graph = None

        if not self.global_config.force_index_from_scratch:
            if os.path.exists(self._graph_pickle_filename):
                preloaded_graph = ig.Graph.Read_Pickle(self._graph_pickle_filename)

        if preloaded_graph is None:
            return ig.Graph(directed=self.global_config.is_directed_graph)
        else:
            logger.info(
                f"Loaded graph from {self._graph_pickle_filename} with {preloaded_graph.vcount()} nodes, {preloaded_graph.ecount()} edges"
            )
            return preloaded_graph

    def index(self, docs: List[str]):
        """
        Indexes documents using fragment-based AtomicRAG framework.

        Fragment-centric architecture:
        1. Extract fragments, entities, and triples from documents via OpenIE
        2. Encode fragments (primary retrieval unit), entities, and facts
        3. Build graph with entity-fragment connections and entity-entity relations

        Parameters:
            docs: List[str] - Raw documents to be indexed
        """
        indexing_start_time = time.time()
        self.stats.num_documents = len(docs)

        logger.info(f"Indexing {len(docs)} Documents (Fragment-based)")
        logger.info(f"Performing OpenIE")

        # Create doc_id -> doc_text mapping
        doc_ids = [compute_mdhash_id(doc, 'doc-') for doc in docs]
        doc_to_rows = {doc_id: {'hash_id': doc_id, 'content': doc} for doc_id, doc in zip(doc_ids, docs)}

        # Load existing OpenIE info and identify new docs to process
        all_openie_info, doc_keys_to_process = self.load_existing_openie(doc_to_rows.keys())
        new_openie_rows = {k: doc_to_rows[k] for k in doc_keys_to_process}

        # Perform batch OpenIE if there are new docs to process
        if len(doc_keys_to_process) > 0:
            # Clear metadata list before OpenIE
            self.openie.metadata_list = []

            if self.global_config.unified_extraction:
                new_ner_results_dict, new_triple_results_dict, new_fragment_results_dict = self.openie.batch_unified_openie(new_openie_rows)
            else:
                new_ner_results_dict, new_triple_results_dict, new_fragment_results_dict = self.openie.batch_openie(new_openie_rows)

            # Collect OpenIE metadata for statistics
            self._update_token_stats(self.openie.metadata_list, 'openie')

            # Log OpenIE token statistics
            logger.info(f"📊 OpenIE Stage - Tokens: prompt={self.stats.openie_prompt_tokens:,}, "
                       f"completion={self.stats.openie_completion_tokens:,}, "
                       f"total={self.stats.openie_prompt_tokens + self.stats.openie_completion_tokens:,}")

            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict, new_fragment_results_dict)

        # Save OpenIE results if configured
        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        # Reformat OpenIE results
        ner_results_dict, triple_results_dict, fragment_results_dict = reformat_openie_results(all_openie_info)

        # Verify consistency
        assert len(doc_to_rows) == len(ner_results_dict) == len(triple_results_dict) == len(fragment_results_dict)

        # Get all doc IDs
        doc_ids_list = list(doc_to_rows.keys())

        # Process triples for each doc
        doc_triples = [[text_processing(t) for t in triple_results_dict[doc_id].triples] for doc_id in doc_ids_list]

        # Extract entity nodes (used for graph construction, not primary retrieval)
        entity_nodes, doc_triple_entities = extract_entity_nodes(doc_triples)

        # Flatten all triples (facts)
        facts = flatten_facts(doc_triples)

        # Get fragments and their entities
        doc_fragments = [fragment_results_dict[doc_id].knowledge_fragments for doc_id in doc_ids_list]
        doc_fragment_entities = [fragment_results_dict[doc_id].fragment_entities for doc_id in doc_ids_list]

        # Flatten all fragments (primary retrieval unit)
        all_fragments = [fragment for fragments in doc_fragments for fragment in fragments]

        # Encode entities (for graph connections)
        logger.info(f"Encoding {len(entity_nodes)} Entities")
        self.entity_embedding_store.insert_strings(entity_nodes)

        # Fact embeddings removed - only triples are used for graph construction (entity-entity edges)
        # Encode fragments (primary retrieval unit)
        logger.info(f"Encoding {len(all_fragments)} Knowledge Fragments")
        self.hierarchical_fragment_store.insert_strings(all_fragments)

        # Build graph structure
        logger.info(f"Constructing Fragment-based Graph")

        # Initialize node relationship stats and entity-fragment mapping
        self.node_to_node_stats = {}
        self.ent_node_to_fragment_ids = {}

        # Add entity-entity edges based on triples
        self.add_fact_edges(doc_ids_list, doc_triples)

        # Add fragment-entity edges using precise mappings
        num_new_fragments = self.build_hierarchical_fragment_connections(doc_ids_list, doc_fragments, doc_fragment_entities)

        # Build doc -> fragments mapping for traceability
        self._build_doc_fragment_mapping(doc_ids_list, doc_fragments)

        # If new fragments were added, complete graph construction
        if num_new_fragments > 0:
            logger.info(f"Found {num_new_fragments} new knowledge fragments to save into graph.")
            self.add_synonymy_edges()
            self.augment_graph()
            self.save_igraph()

        # Record indexing time
        indexing_end_time = time.time()
        self.stats.indexing_time = indexing_end_time - indexing_start_time
        logger.info(f"🏗️  Indexing Stage - Time: {self.stats.indexing_time:.2f}s")

    def _build_doc_fragment_mapping(self, doc_ids: List[str], doc_fragments: List[List[str]]):
        """Build mapping from original documents to their fragments for traceability"""
        for doc_id, fragments in zip(doc_ids, doc_fragments):
            fragment_ids = [compute_mdhash_id(fragment, 'fragment-') for fragment in fragments]
            self.doc_to_fragment_ids[doc_id] = fragment_ids

    async def retrieve_async(self,
                             queries: List[str],
                             num_to_retrieve: int = None,
                             decomposition_expand_factor: float = 1.0,
                             question_types: Optional[List[str]] = None,
                             max_workers: int = None) -> List[QuerySolution]:
        """
        Async-first concurrent retrieval.

        The flow matches the synchronous implementation but leverages asyncio to
        orchestrate thread pools and downstream async filters without blocking
        the caller's event loop.
        """
        retrieve_start_time = time.time()

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        if question_types is None:
            question_types = [None] * len(queries)

        # Step 1: Process all queries for decomposition CONCURRENTLY using async
        logger.info(f"Processing {len(queries)} queries with CONCURRENT decomposition (asyncio)")

        # Use full configured max_concurrency so evaluation/decomposition share the same limit
        decomp_max_concurrent = self.global_config.max_concurrency

        # Start decomposition timing
        decomp_start_time = time.time()

        decomposition_results = await self.query_decomposer.process_queries_batch_async(
            questions=queries,
            question_types=question_types,
            max_concurrent=decomp_max_concurrent
        )

        # Record decomposition time and collect metadata (use += for multi-batch accumulation)
        decomp_end_time = time.time()
        self.stats.query_decomposition_time += decomp_end_time - decomp_start_time

        # Collect decomposition metadata
        if hasattr(self.query_decomposer, 'metadata_list'):
            self._update_token_stats(self.query_decomposer.metadata_list, 'decomposition')
            self.query_decomposer.metadata_list = []  # Clear for next call

        # Log decomposition statistics
        logger.info(f"⏱️  Query Decomposition - Time: {decomp_end_time - decomp_start_time:.2f}s, "
                   f"Tokens: prompt={self.stats.decomposition_prompt_tokens:,}, "
                   f"completion={self.stats.decomposition_completion_tokens:,}, "
                   f"total={self.stats.decomposition_prompt_tokens + self.stats.decomposition_completion_tokens:,}")

        # Step 2: Collect all queries to retrieve (original + sub-questions)
        all_queries_to_retrieve = []
        query_mapping = []  # (original_query_idx, is_sub_question, sub_question_id)

        for q_idx, decomp_result in enumerate(decomposition_results):
            if decomp_result.is_decomposed and len(decomp_result.sub_questions) > 0:
                # Add original question first
                all_queries_to_retrieve.append(decomp_result.original_question)
                query_mapping.append((q_idx, False, -1))

                # Then add all sub-questions
                for sub_q in decomp_result.sub_questions:
                    all_queries_to_retrieve.append(sub_q.question)
                    query_mapping.append((q_idx, True, sub_q.id))
            else:
                # Add original question only
                all_queries_to_retrieve.append(decomp_result.original_question)
                query_mapping.append((q_idx, False, -1))

        # Step 3: CONCURRENT retrieval of all queries using ThreadPoolExecutor
        # Get embeddings for all queries (can be batched efficiently)
        self.get_query_embeddings(all_queries_to_retrieve)

        # Determine max workers based on CPU cores
        # CRITICAL: Limit based on CPU count to prevent lock contention on self._graph_lock in run_ppr()
        # python-igraph is NOT thread-safe, so all PPR operations serialize on one lock
        #
        # SERIAL MODE (max_workers=1): Most stable, avoids all lock contention
        # Use serial mode to prevent any possibility of hanging due to lock contention

        if max_workers is None:
            config_workers = getattr(self.global_config, "retrieval_max_workers", None)
            if isinstance(config_workers, int) and config_workers > 0:
                max_workers = config_workers
            else:
                max_workers = 1  # Serial mode by default for maximum stability

        # 先抽取实体，再预计算实体 embedding，避免检索阶段被阻塞
        # 使用同步版本，在独立的 event loop 中运行，与 NER 阶段保持一致
        self.prefetch_query_entities(all_queries_to_retrieve)
        self.prefetch_query_entity_embeddings(all_queries_to_retrieve)

        logger.info(f"Retrieving {len(all_queries_to_retrieve)} queries (max_workers={max_workers})")

        # Define single retrieval function
        def retrieve_single_query(query_idx_and_text):
            query_idx, query_text = query_idx_and_text
            # Use graph search with DPR + PPR (no fact embeddings needed)
            sorted_doc_ids, sorted_doc_scores = self.graph_search_with_fact_entities(
                query=query_text,
                passage_node_weight=self.global_config.passage_node_weight
            )

            top_k_docs = [self.hierarchical_fragment_store.get_row(self.fragment_node_keys[idx])["content"]
                         for idx in sorted_doc_ids[:num_to_retrieve]]

            return {
                "query_idx": query_idx,
                "query": query_text,
                "docs": top_k_docs,
                "scores": sorted_doc_scores[:num_to_retrieve]
            }

        loop = asyncio.get_running_loop()
        indexed_queries = list(enumerate(all_queries_to_retrieve))
        executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
            futures = [
                loop.run_in_executor(executor, retrieve_single_query, indexed_query)
                for indexed_query in indexed_queries
            ]
            all_retrieval_results = []
            progress_iter = tqdm(
                asyncio.as_completed(futures),
                total=len(futures),
                desc="Knowledge Retrieval",
                leave=True,
                mininterval=0.5,
                disable=False
            )
            processed = 0
            total = len(futures)
            for future in progress_iter:
                all_retrieval_results.append(await future)
                processed += 1
        finally:
            executor.shutdown(wait=True)

        # Sort results by query index to maintain order
        all_retrieval_results = sorted(all_retrieval_results, key=lambda x: x["query_idx"])

        # Filter each query's retrieved fragments before merging.
        per_query_filter_requests = [
            FragmentFilterRequest(
                request_id=result["query_idx"],
                question=result["query"],
                fragments=result["docs"]
            )
            for result in all_retrieval_results
            if result["docs"]
        ]

        if per_query_filter_requests:
            logger.info(f"Pre-filtering fragments for {len(per_query_filter_requests)} queries")

            filter_start_time = time.time()
            batch_results = await self.fragment_filter.filter_fragments_batch_async(
                requests=per_query_filter_requests
            )
            filter_end_time = time.time()
            self.stats.fragment_filter_time += filter_end_time - filter_start_time

            if hasattr(self.fragment_filter, 'metadata_list'):
                self._update_token_stats(self.fragment_filter.metadata_list, 'filter')
                self.fragment_filter.metadata_list = []

            logger.info(f"🔍 Fragment Filter (per-query) - Time: {filter_end_time - filter_start_time:.2f}s, "
                       f"Tokens: prompt={self.stats.filter_prompt_tokens:,}, "
                       f"completion={self.stats.filter_completion_tokens:,}, "
                       f"total={self.stats.filter_prompt_tokens + self.stats.filter_completion_tokens:,}")

            # Apply filtered fragments back to results
            retrieval_result_map = {r["query_idx"]: r for r in all_retrieval_results}
            for request in per_query_filter_requests:
                filtered_docs, kept_indices = batch_results.get(
                    request.request_id,
                    (request.fragments, list(range(len(request.fragments))))
                )
                result_ref = retrieval_result_map.get(request.request_id)
                if result_ref is None:
                    continue
                result_ref["docs"] = filtered_docs
                if kept_indices:
                    result_ref["scores"] = np.array([result_ref["scores"][i] for i in kept_indices])
                else:
                    result_ref["scores"] = np.array([])

        # Step 4: Merge results for decomposed queries (same as sequential version)
        query_solution_data: List[Dict[str, Any]] = []

        for q_idx, decomp_result in enumerate(decomposition_results):
            if decomp_result.is_decomposed and len(decomp_result.sub_questions) > 0:
                # Find original and sub-question results
                original_result = None
                sub_results = []
                sub_contexts = []

                for retrieve_idx, (orig_idx, is_sub, sub_id) in enumerate(query_mapping):
                    if orig_idx == q_idx:
                        result = all_retrieval_results[retrieve_idx]
                        if not is_sub:
                            original_result = result
                        else:
                            sub_results.append(result)
                            sub_contexts.append(result["docs"])

                            # Update sub_question with retrieved docs
                            for sub_q in decomp_result.sub_questions:
                                if sub_q.id == sub_id:
                                    sub_q.docs = result["docs"]

                # Merge contexts with score-aware deduplication so sub-questions
                # keep their highest-scoring fragments before QA truncation.
                doc_info: Dict[str, Dict[str, Any]] = {}
                insertion_order = 0

                def consider_docs(docs: List[str], scores, source_tag: str):
                    nonlocal insertion_order
                    if scores is None:
                        scores = []
                    for idx, doc in enumerate(docs):
                        score_val = float(scores[idx]) if idx < len(scores) else 0.0
                        existing = doc_info.get(doc)
                        if existing is None or score_val > existing["score"]:
                            doc_info[doc] = {
                                "score": score_val,
                                "order": insertion_order,
                                "source": source_tag
                            }
                        insertion_order += 1

                if original_result:
                    original_docs = original_result["docs"]
                    consider_docs(original_docs, original_result.get("scores", []), "original")
                else:
                    original_docs = []

                # Add sub-question docs with their scores
                for sub_idx, sub_result in enumerate(sub_results):
                    consider_docs(
                        sub_result.get("docs", []),
                        sub_result.get("scores", []),
                        f"sub_{sub_idx}"
                    )

                # Sort by score descending, then by insertion order to keep determinism
                sorted_docs = sorted(
                    doc_info.items(),
                    key=lambda item: (-item[1]["score"], item[1]["order"])
                )
                final_docs = [doc for doc, meta in sorted_docs]
                final_scores = np.array([meta["score"] for _, meta in sorted_docs])

                # Store metadata
                decomp_result.sub_contexts = sub_contexts
                decomp_result.original_docs = original_docs
                decomp_result.merged_context = final_docs

                query_solution_data.append({
                    "question": decomp_result.original_question,
                    "docs": final_docs,
                    "scores": np.array(final_scores),
                    "metadata": decomp_result.to_dict()
                })
            else:
                # Non-decomposed query
                for retrieve_idx, (orig_idx, is_sub, sub_id) in enumerate(query_mapping):
                    if orig_idx == q_idx and not is_sub:
                        result = all_retrieval_results[retrieve_idx]

                        final_docs = result["docs"]
                        final_scores = np.array(result["scores"])

                        query_solution_data.append({
                            "question": decomp_result.original_question,
                            "docs": final_docs,
                            "scores": np.array(final_scores),
                            "metadata": decomp_result.to_dict()
                        })
                        break

        retrieval_results: List[QuerySolution] = []
        for entry in query_solution_data:
            query_solution = QuerySolution(
                question=entry["question"],
                docs=entry["docs"]
            )
            query_solution.decomposition_metadata = entry["metadata"]
            retrieval_results.append(query_solution)

        retrieve_end_time = time.time()

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"🔎 Retrieval Stage - Total Time: {retrieve_end_time - retrieve_start_time:.2f}s, "
                   f"PPR: {self.ppr_time:.2f}s")

        # Log decomposition and concurrency statistics
        num_decomposed = sum(1 for dr in decomposition_results if dr.is_decomposed)
        if num_decomposed > 0:
            logger.info(f"Decomposition Stats: {num_decomposed}/{len(queries)} queries decomposed")
            avg_sub_questions = np.mean([len(dr.sub_questions) for dr in decomposition_results if dr.is_decomposed])
            logger.info(f"Average sub-questions: {avg_sub_questions:.2f}")

        total_retrievals = len(all_queries_to_retrieve)
        sequential_time_estimate = (retrieve_end_time - retrieve_start_time) * total_retrievals / max_workers
        speedup = sequential_time_estimate / (retrieve_end_time - retrieve_start_time)
        logger.info(f"Concurrent Performance: {total_retrievals} retrievals, "
                   f"~{speedup:.2f}x speedup (estimated)")

        return retrieval_results

    def retrieve(self,
                 queries: List[str],
                 num_to_retrieve: int = None,
                 decomposition_expand_factor: float = 1.0,
                 question_types: Optional[List[str]] = None,
                 max_workers: int = None) -> List[QuerySolution]:
        """
        Synchronous wrapper for `retrieve_async` (backward compatible entry point).
        """
        return self._run_async(self.retrieve_async(
            queries=queries,
            num_to_retrieve=num_to_retrieve,
            decomposition_expand_factor=decomposition_expand_factor,
            question_types=question_types,
            max_workers=max_workers
        ))

    async def rag_qa_async(self,
                           queries: List[str | QuerySolution],
                           gold_answers: List[List[str]] = None
                           ) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """
        Async version of retrieval-augmented QA using the AtomicRAG framework.
        """
        if not isinstance(queries[0], QuerySolution):
            queries = await self.retrieve_async(queries=queries)

        queries_solutions, all_response_message, all_metadata = await self.qa_async(queries)

        if gold_answers is not None:
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])

        return queries_solutions, all_response_message, all_metadata

    async def qa_async(self, queries: List[QuerySolution]) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """
        Async QA inference over retrieved contexts.
        """
        qa_start_time = time.time()
        all_qa_messages = []

        for query_solution in tqdm(queries, desc="Collecting QA prompts"):
            retrieved_passages = query_solution.docs[:self.global_config.qa_top_k]

            prompt_user = ''
            for passage in retrieved_passages:
                prompt_user += f'Wikipedia Title: {passage}\n\n'
            prompt_user += 'Question: ' + query_solution.question + '\nThought: '

            prompt_dataset_name = None
            prompt_override = getattr(self.global_config, "qa_prompt_template", None)
            if prompt_override:
                override_template = f'rag_qa_{prompt_override}'
                if self.prompt_template_manager.is_template_name_valid(name=override_template):
                    prompt_dataset_name = prompt_override
                else:
                    logger.warning(f"qa_prompt_template '{prompt_override}' not found. Falling back to dataset mapping.")

            dataset_name = getattr(self.global_config, "dataset", None)
            if (
                prompt_dataset_name is None
                and dataset_name
                and self.prompt_template_manager.is_template_name_valid(name=f'rag_qa_{dataset_name}')
            ):
                prompt_dataset_name = dataset_name

            if prompt_dataset_name is None:
                logger.debug(
                    f"rag_qa_{dataset_name if dataset_name else 'None'} does not have a customized prompt template. Using ABSTRACT default prompt instead.")
                prompt_dataset_name = 'abstract'
            all_qa_messages.append(
                self.prompt_template_manager.render(name=f'rag_qa_{prompt_dataset_name}', prompt_user=prompt_user))

        if not all_qa_messages:
            all_response_message, all_metadata = [], []
        else:
            worker_limit = max(1, getattr(self.global_config, "max_concurrency", 1))
            semaphore = asyncio.Semaphore(worker_limit)
            results: List[Optional[Tuple[str, dict, bool]]] = [None] * len(all_qa_messages)
            task_timeout = 60.0  # 60s timeout per task
            logger.info(f"QA Reading: {len(all_qa_messages)} tasks with {worker_limit} workers")

            async def _run_single(idx: int, qa_messages):
                async with semaphore:
                    try:
                        return idx, await asyncio.wait_for(
                            self.llm_model.ainfer(qa_messages),
                            timeout=task_timeout
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"QA task {idx} timed out")
                        return idx, ("", {}, False)

            tasks = [
                asyncio.create_task(_run_single(idx, qa_messages))
                for idx, qa_messages in enumerate(all_qa_messages)
            ]

            progress_iter = tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="QA Reading",
                leave=True,
                mininterval=0.5,
                disable=False
            )
            processed = 0
            total = len(tasks)
            for task in progress_iter:
                idx, result = await task
                results[idx] = result
                processed += 1

            all_response_message, all_metadata, _ = zip(*results) if results else ([], [], [])
            all_response_message = list(all_response_message)
            all_metadata = list(all_metadata)

        queries_solutions = []
        for query_solution_idx, query_solution in tqdm(enumerate(queries), desc="Extraction Answers from LLM Response"):
            response_content = all_response_message[query_solution_idx]
            try:
                pred_ans = response_content.split('Answer:')[1].strip()
            except Exception as e:
                logger.warning(f"Error in parsing the answer from the raw LLM QA inference response: {str(e)}!")
                pred_ans = response_content

            query_solution.answer = pred_ans
            queries_solutions.append(query_solution)

        # Collect QA token statistics
        self._update_token_stats(all_metadata, 'qa')

        # Record QA time (use += for multi-batch accumulation)
        qa_end_time = time.time()
        self.stats.qa_time += qa_end_time - qa_start_time

        # Log QA statistics
        logger.info(f"💬 QA Stage - Time: {qa_end_time - qa_start_time:.2f}s, "
                   f"Tokens: prompt={self.stats.qa_prompt_tokens:,}, "
                   f"completion={self.stats.qa_completion_tokens:,}, "
                   f"total={self.stats.qa_prompt_tokens + self.stats.qa_completion_tokens:,}")

        return queries_solutions, all_response_message, all_metadata

    def qa(self, queries: List[QuerySolution]) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """
        Synchronous wrapper for `qa_async`.
        """
        return self._run_async(self.qa_async(queries))

    def add_fact_edges(self, doc_ids: List[str], doc_triples: List[Tuple]):
        """
        Add entity-entity edges from triples to the graph (fragment-based).

        Creates bidirectional edges between entities that appear in the same triple.
        Note: This method no longer tracks entity-to-doc mappings as we use
        entity-to-fragment mappings instead.

        Parameters:
            doc_ids: List[str] - Document identifiers (for logging/tracking)
            doc_triples: List[Tuple] - Triples extracted from each document
        """
        if "name" in self.graph.vs:
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        logger.info(f"Adding OpenIE triples to graph (fragment-based)")

        for doc_key, triples in tqdm(zip(doc_ids, doc_triples), desc="Processing triples"):
            # Skip if this doc was already processed
            if doc_key in current_graph_nodes:
                continue

            for triple in triples:
                triple = tuple(triple)

                # Get entity node IDs
                subject_key = compute_mdhash_id(content=triple[0], prefix="entity-")
                object_key = compute_mdhash_id(content=triple[2], prefix="entity-")

                # Add bidirectional edges between subject and object
                self.node_to_node_stats[(subject_key, object_key)] = \
                    self.node_to_node_stats.get((subject_key, object_key), 0.0) + 1
                self.node_to_node_stats[(object_key, subject_key)] = \
                    self.node_to_node_stats.get((object_key, subject_key), 0.0) + 1

    def build_hierarchical_fragment_connections(self, doc_ids: List[str], doc_fragments: List[List[str]], doc_fragment_entities: List[List[List[str]]]):
        """
        Build fragment-entity edges in the graph using precise mappings (fragment-based architecture).

        This creates bidirectional connections between knowledge fragments and their specific entities.
        The hierarchical structure enables better multi-hop reasoning and associative retrieval.

        Parameters:
            doc_ids: List[str] - Document identifiers (for tracking source)
            doc_fragments: List[List[str]] - Knowledge fragments extracted from each document
            doc_fragment_entities: List[List[List[str]]] - For each doc, for each fragment, its specific entities

        Returns:
            int - Number of new knowledge fragment nodes added to the graph
        """

        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        num_new_fragments = 0

        logger.info(f"Connecting knowledge fragment nodes to entity nodes using precise fragment-entity mappings (fragment-based).")

        # For each document, connect its fragments to their specific entities
        for idx, doc_id in tqdm(enumerate(doc_ids), desc="Building fragment-entity connections"):
            fragments_in_doc = doc_fragments[idx]
            fragment_entities_in_doc = doc_fragment_entities[idx]

            # Ensure fragment_entities list matches fragments list length
            if len(fragment_entities_in_doc) != len(fragments_in_doc):
                # Pad with empty lists if needed
                while len(fragment_entities_in_doc) < len(fragments_in_doc):
                    fragment_entities_in_doc.append([])

            # For each fragment in this doc, create connections to its specific entities
            for fragment_idx, fragment in enumerate(fragments_in_doc):
                # Generate unique fragment ID
                fragment_key = compute_mdhash_id(content=fragment, prefix="fragment-")

                if fragment_key not in current_graph_nodes:
                    # Get the specific entities for this fragment
                    entities_for_this_fragment = (
                        fragment_entities_in_doc[fragment_idx]
                        if fragment_idx < len(fragment_entities_in_doc)
                        else []
                    )

                    # Connect fragment to only its specific entities
                    for entity in entities_for_this_fragment:
                        entity_key = compute_mdhash_id(content=entity, prefix="entity-")

                        # Create bidirectional edge between fragment and entity
                        self.node_to_node_stats[(fragment_key, entity_key)] = 1.0
                        self.node_to_node_stats[(entity_key, fragment_key)] = 1.0

                        # Track which fragments contain which entities
                        if entity_key not in self.ent_node_to_fragment_ids:
                            self.ent_node_to_fragment_ids[entity_key] = set()
                        self.ent_node_to_fragment_ids[entity_key].add(fragment_key)

                    num_new_fragments += 1

        return num_new_fragments

    def add_synonymy_edges(self):
        """
        Adds synonymy edges between entities and (optionally) fragments
        using embedding similarity.
        """
        logger.info("Expanding graph with synonymy edges")

        entity_keys = list(self.entity_embedding_store.get_all_ids())
        entity_edges = self._add_synonym_edges_for_nodes(
            node_label="entity",
            store=self.entity_embedding_store,
            node_keys=entity_keys,
            topk=self.global_config.synonymy_edge_topk,
            query_batch_size=self.global_config.synonymy_edge_query_batch_size,
            key_batch_size=self.global_config.synonymy_edge_key_batch_size,
            sim_threshold=self.global_config.synonymy_edge_sim_threshold,
            min_clean_length=3
        )

        logger.info(f"Synonymy edge summary: entity_edges={entity_edges}")

    def _add_synonym_edges_for_nodes(self,
                                     *,
                                     node_label: str,
                                     store: EmbeddingStore,
                                     node_keys: List[str],
                                     topk: int,
                                     query_batch_size: int,
                                     key_batch_size: int,
                                     sim_threshold: float,
                                     min_clean_length: int = 0) -> int:
        """
        Helper to add synonym edges for a specific embedding store.
        """
        if not node_keys or topk <= 0:
            logger.info(f"No {node_label} nodes available for synonym edges.")
            return 0

        node_rows = store.get_all_id_to_rows()
        node_embeddings = store.get_embeddings(node_keys)

        logger.info(f"Performing KNN retrieval for {len(node_keys)} {node_label} nodes.")
        knn_results = retrieve_knn(
            query_ids=node_keys,
            key_ids=node_keys,
            query_vecs=node_embeddings,
            key_vecs=node_embeddings,
            k=topk,
            query_batch_size=query_batch_size,
            key_batch_size=key_batch_size
        )

        added_edges = 0
        max_neighbors = min(topk, 100)
        desc = f"{node_label.capitalize()} synonym search"
        for node_key in tqdm(node_keys, total=len(node_keys), desc=desc):
            base_entry = node_rows.get(node_key, {})
            base_text = base_entry.get("content", "")
            if min_clean_length and len(re.sub('[^A-Za-z0-9]', '', base_text)) <= min_clean_length:
                continue

            nn_tuple = knn_results.get(node_key)
            if not nn_tuple:
                continue

            neighbor_ids, neighbor_scores = nn_tuple
            neighbor_count = 0
            for nn, score in zip(neighbor_ids, neighbor_scores):
                if score < sim_threshold or neighbor_count >= max_neighbors:
                    break
                if nn == node_key:
                    continue

                neighbor_entry = node_rows.get(nn, {})
                if not neighbor_entry.get("content"):
                    continue

                self.node_to_node_stats[(node_key, nn)] = score
                added_edges += 1
                neighbor_count += 1

        logger.info(f"Added {added_edges} {node_label} synonym edges.")
        return added_edges

    def load_existing_openie(self, chunk_keys: List[str]) -> Tuple[List[dict], Set[str]]:
        """
        Loads existing OpenIE results from the specified file if it exists and combines
        them with new content while standardizing indices. If the file does not exist or
        is configured to be re-initialized from scratch with the flag `force_openie_from_scratch`,
        it prepares new entries for processing.

        Args:
            chunk_keys (List[str]): A list of chunk keys that represent identifiers
                                     for the content to be processed.

        Returns:
            Tuple[List[dict], Set[str]]: A tuple where the first element is the existing OpenIE
                                         information (if any) loaded from the file, and the
                                         second element is a set of chunk keys that still need to
                                         be saved or processed.
        """

        # combine openie_results with contents already in file, if file exists
        chunk_keys_to_save = set()

        if not self.global_config.force_openie_from_scratch and os.path.isfile(self.openie_results_path):
            with open(self.openie_results_path, "r", encoding="utf-8") as f:
                openie_results = json.load(f)
            all_openie_info = openie_results.get('docs', [])

            # Standardize indices: legacy caches may use sequential or "chunk-" prefixed hashes.
            normalized_openie_info = []
            for openie_info in all_openie_info:
                idx = openie_info.get('idx')
                passage = openie_info.get('passage', '')
                if not idx:
                    idx = compute_mdhash_id(passage, 'doc-')
                elif idx.startswith('chunk-'):
                    idx = compute_mdhash_id(passage, 'doc-')
                openie_info['idx'] = idx
                normalized_openie_info.append(openie_info)

            all_openie_info = normalized_openie_info

            existing_openie_keys = set(info['idx'] for info in all_openie_info)

            for chunk_key in chunk_keys:
                if chunk_key not in existing_openie_keys:
                    chunk_keys_to_save.add(chunk_key)
        else:
            all_openie_info = []
            chunk_keys_to_save = chunk_keys

        return all_openie_info, chunk_keys_to_save

    def merge_openie_results(self,
                             all_openie_info: List[dict],
                             chunks_to_save: Dict[str, dict],
                             ner_results_dict: Dict[str, NerRawOutput],
                             triple_results_dict: Dict[str, TripleRawOutput],
                             fragment_results_dict: Dict[str, KnowledgeFragmentRawOutput]) -> List[dict]:
        """
        Merges OpenIE extraction results with corresponding passage and metadata.

        This function integrates the OpenIE extraction results, including named-entity
        recognition (NER) entities and triples, with their respective text passages
        using the provided chunk keys. The resulting merged data is appended to
        the `all_openie_info` list containing dictionaries with combined and organized
        data for further processing or storage.

        Parameters:
            all_openie_info (List[dict]): A list to hold dictionaries of merged OpenIE
                results and metadata for all chunks.
            chunks_to_save (Dict[str, dict]): A dict of chunk identifiers (keys) to process
                and merge OpenIE results to dictionaries with `hash_id` and `content` keys.
            ner_results_dict (Dict[str, NerRawOutput]): A dictionary mapping chunk keys
                to their corresponding NER extraction results.
            triple_results_dict (Dict[str, TripleRawOutput]): A dictionary mapping chunk
                keys to their corresponding OpenIE triple extraction results.

        Returns:
            List[dict]: The `all_openie_info` list containing dictionaries with merged
            OpenIE results, metadata, and the passage content for each chunk.

        """

        for chunk_key, row in chunks_to_save.items():
            passage = row['content']
            chunk_openie_info = {'idx': chunk_key, 'passage': passage,
                                 'extracted_entities': ner_results_dict[chunk_key].unique_entities,
                                 'extracted_triples': triple_results_dict[chunk_key].triples,
                                 'extracted_knowledge_fragments': fragment_results_dict[chunk_key].knowledge_fragments,
                                 'extracted_fragment_entities': fragment_results_dict[chunk_key].fragment_entities}
            all_openie_info.append(chunk_openie_info)

        return all_openie_info

    def save_openie_results(self, all_openie_info: List[dict]):
        """
        Computes statistics on extracted entities from OpenIE results and saves the aggregated data in a
        JSON file. The function calculates the average character and word lengths of the extracted entities
        and writes them along with the provided OpenIE information to a file.

        Parameters:
            all_openie_info : List[dict]
                List of dictionaries, where each dictionary represents information from OpenIE, including
                extracted entities.
        """

        sum_phrase_chars = sum([len(e) for chunk in all_openie_info for e in chunk['extracted_entities']])
        sum_phrase_words = sum([len(e.split()) for chunk in all_openie_info for e in chunk['extracted_entities']])
        num_phrases = sum([len(chunk['extracted_entities']) for chunk in all_openie_info])

        if len(all_openie_info) > 0:
            # Avoid division by zero if there are no phrases
            if num_phrases > 0:
                avg_ent_chars = round(sum_phrase_chars / num_phrases, 4)
                avg_ent_words = round(sum_phrase_words / num_phrases, 4)
            else:
                avg_ent_chars = 0
                avg_ent_words = 0

            openie_dict = {
                'docs': all_openie_info,
                'avg_ent_chars': avg_ent_chars,
                'avg_ent_words': avg_ent_words
            }

            with open(self.openie_results_path, 'w') as f:
                json.dump(openie_dict, f)
            logger.info(f"OpenIE results saved to {self.openie_results_path}")

    def augment_graph(self):
        """
        Provides utility functions to augment a graph by adding new nodes and edges.
        It ensures that the graph structure is extended to include additional components,
        and logs the completion status along with printing the updated graph information.
        """

        self.add_new_nodes()
        self.add_new_edges()

        logger.info(f"Graph construction completed!")
        print(self.get_graph_info())

    def add_new_nodes(self):
        """
        Adds new nodes to the graph from entity and passage embedding stores based on their attributes.

        This method identifies and adds new nodes to the graph by comparing existing nodes
        in the graph and nodes retrieved from the entity embedding store and the passage
        embedding store. The method checks attributes and ensures no duplicates are added.
        New nodes are prepared and added in bulk to optimize graph updates.
        """

        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()}

        entity_to_row = self.entity_embedding_store.get_all_id_to_rows()
        fragment_to_row = self.hierarchical_fragment_store.get_all_id_to_rows()

        node_to_rows = entity_to_row
        node_to_rows.update(fragment_to_row)

        new_nodes = {}
        for node_id, node in node_to_rows.items():
            node['name'] = node_id
            if node_id not in existing_nodes:
                for k, v in node.items():
                    if k not in new_nodes:
                        new_nodes[k] = []
                    new_nodes[k].append(v)

        if len(new_nodes) > 0:
            self.graph.add_vertices(n=len(next(iter(new_nodes.values()))), attributes=new_nodes)

    def add_new_edges(self):
        """
        Processes edges from `node_to_node_stats` to add them into a graph object while
        managing adjacency lists, validating edges, and logging invalid edge cases.
        """

        graph_adj_list = defaultdict(dict)
        graph_inverse_adj_list = defaultdict(dict)
        edge_source_node_keys = []
        edge_target_node_keys = []
        edge_metadata = []
        for edge, weight in self.node_to_node_stats.items():
            if edge[0] == edge[1]: continue
            graph_adj_list[edge[0]][edge[1]] = weight
            graph_inverse_adj_list[edge[1]][edge[0]] = weight

            edge_source_node_keys.append(edge[0])
            edge_target_node_keys.append(edge[1])
            edge_metadata.append({
                "weight": weight
            })

        valid_edges, valid_weights = [], {"weight": []}
        current_node_ids = set(self.graph.vs["name"])
        for source_node_id, target_node_id, edge_d in zip(edge_source_node_keys, edge_target_node_keys, edge_metadata):
            if source_node_id in current_node_ids and target_node_id in current_node_ids:
                valid_edges.append((source_node_id, target_node_id))
                weight = edge_d.get("weight", 1.0)
                valid_weights["weight"].append(weight)
        self.graph.add_edges(
            valid_edges,
            attributes=valid_weights
        )

    def save_igraph(self):
        logger.info(
            f"Writing graph with {len(self.graph.vs())} nodes, {len(self.graph.es())} edges"
        )
        self.graph.write_pickle(self._graph_pickle_filename)
        logger.info(f"Saving graph completed!")

    def get_graph_info(self) -> Dict:
        """
        Obtains detailed information about the graph such as the number of nodes,
        triples, and their classifications.

        This method calculates various statistics about the graph based on the
        stores and node-to-node relationships, including counts of phrase and
        passage nodes, total nodes, extracted triples, triples involving passage
        nodes, synonymy triples, and total triples.

        Returns:
            Dict
                A dictionary containing the following keys and their respective values:
                - num_phrase_nodes: The number of unique phrase nodes.
                - num_passage_nodes: The number of unique passage nodes.
                - num_total_nodes: The total number of nodes (sum of phrase and passage nodes).
                - num_extracted_triples: The number of unique extracted triples.
                - num_triples_with_passage_node: The number of triples involving at least one
                  passage node.
                - num_synonymy_triples: The number of synonymy triples (distinct from extracted
                  triples and those with passage nodes).
                - num_total_triples: The total number of triples.
        """
        graph_info = {}

        # get # of phrase nodes
        phrase_nodes_keys = self.entity_embedding_store.get_all_ids()
        graph_info["num_phrase_nodes"] = len(set(phrase_nodes_keys))

        # get # of knowledge fragment nodes
        fragment_nodes_keys = self.hierarchical_fragment_store.get_all_ids()
        graph_info["num_fragment_nodes"] = len(set(fragment_nodes_keys))

        # get # of total nodes
        graph_info["num_total_nodes"] = graph_info["num_phrase_nodes"] + graph_info["num_fragment_nodes"]

        # get # of extracted triples (counted from entity-entity edges)
        # Note: Each triple creates 2 edges (bidirectional), but we count entity-entity pairs
        entity_nodes_set = set(phrase_nodes_keys)
        num_entity_entity_edges = sum(
            1 for node_pair in self.node_to_node_stats
            if node_pair[0] in entity_nodes_set and node_pair[1] in entity_nodes_set
        )
        # Divide by 2 since edges are bidirectional
        graph_info["num_extracted_triples"] = num_entity_entity_edges // 2

        num_triples_with_fragment_node = 0
        fragment_nodes_set = set(fragment_nodes_keys)
        num_triples_with_fragment_node = sum(
            1 for node_pair in self.node_to_node_stats
            if node_pair[0] in fragment_nodes_set or node_pair[1] in fragment_nodes_set
        )
        graph_info['num_triples_with_fragment_node'] = num_triples_with_fragment_node

        graph_info['num_synonymy_triples'] = len(self.node_to_node_stats) - graph_info[
            "num_extracted_triples"] - num_triples_with_fragment_node

        # get # of total triples
        graph_info["num_total_triples"] = len(self.node_to_node_stats)

        return graph_info

    def prepare_retrieval_objects(self):
        """
        Prepare in-memory objects for fast retrieval (fragment-based architecture).

        This method loads all necessary embeddings and mappings for fragment-centric retrieval.
        No longer uses chunk-based backward compatibility logic.
        """
        logger.info("Preparing for fast retrieval (fragment-based).")

        # Initialize query embedding cache
        logger.info("Loading keys.")
        # Query embeddings cache (only passage embeddings needed now)
        self.query_to_embedding: Dict = {'passage': {}}
        # Reset per-run query entity caches
        self._query_entity_cache = {}
        self._query_entity_score_cache = {}
        self._query_entity_emb_cache = {}
        self._entity_text_emb_cache = {}

        # Get all node keys
        self.entity_node_keys: List = list(self.entity_embedding_store.get_all_ids())
        self.fragment_node_keys: List = list(self.hierarchical_fragment_store.get_all_ids())
        # fact_node_keys removed - fact embeddings are not used

        logger.info(f"Found {len(self.entity_node_keys)} entities, {len(self.fragment_node_keys)} fragments")

        # Verify graph node count matches our stores
        expected_node_count = len(self.entity_node_keys) + len(self.fragment_node_keys)
        actual_node_count = self.graph.vcount()

        if expected_node_count != actual_node_count:
            logger.warning(f"Graph node count mismatch: expected {expected_node_count}, got {actual_node_count}")
            if actual_node_count == 0 and expected_node_count > 0:
                logger.info(f"Initializing graph with {expected_node_count} nodes")
                self.add_new_nodes()
                self.save_igraph()

        # Create node name -> graph index mapping
        try:
            igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)}
            self.node_name_to_vertex_idx = igraph_name_to_idx

            # Check for missing nodes
            missing_entity_nodes = [node_key for node_key in self.entity_node_keys if node_key not in igraph_name_to_idx]
            missing_fragment_nodes = [node_key for node_key in self.fragment_node_keys if node_key not in igraph_name_to_idx]

            if missing_entity_nodes or missing_fragment_nodes:
                logger.warning(f"Missing nodes in graph: {len(missing_entity_nodes)} entities, {len(missing_fragment_nodes)} fragments")
                self.add_new_nodes()
                self.save_igraph()
                # Update mapping
                igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)}
                self.node_name_to_vertex_idx = igraph_name_to_idx

            # Get graph indices for entities and fragments
            self.entity_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.entity_node_keys]
            self.fragment_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.fragment_node_keys]

        except Exception as e:
            logger.error(f"Error creating node index mapping: {str(e)}")
            self.node_name_to_vertex_idx = {}
            self.entity_node_idxs = []
            self.fragment_node_idxs = []

        # Load all embeddings
        logger.info("Loading embeddings.")
        self.entity_embeddings = np.array(self.entity_embedding_store.get_embeddings(self.entity_node_keys))
        self.fragment_embeddings = np.array(self.hierarchical_fragment_store.get_embeddings(self.fragment_node_keys))
        # fact_embeddings removed - not used in retrieval

        # Load OpenIE info for triple-to-doc mapping (used in graph search)
        all_openie_info, _ = self.load_existing_openie([])

        # Build triple -> docs mapping
        self.proc_triples_to_docs = {}
        for doc in all_openie_info:
            triples = flatten_facts([doc['extracted_triples']])
            for triple in triples:
                if len(triple) == 3:
                    proc_triple = tuple(text_processing(list(triple)))
                    self.proc_triples_to_docs[str(proc_triple)] = \
                        self.proc_triples_to_docs.get(str(proc_triple), set()).union(set([doc['idx']]))

        # Verify entity-to-fragment mapping exists (built during indexing)
        if not self.ent_node_to_fragment_ids:
            logger.warning("Entity-to-fragment mapping is empty. This may affect graph search performance.")
            logger.warning("Ensure that index() was called before retrieval.")

        # Mark as ready
        self.ready_to_retrieve = True
        logger.info("Retrieval preparation complete.")

    def get_query_embeddings(self, queries: List[str] | List[QuerySolution]):
        """
        为给定的查询获取嵌入，并更新内部的查询到嵌入映射。

        本方法会检查每个查询（可以是字符串或 QuerySolution 对象）在内部字典的 'triple' 和 'passage' 键下是否已有嵌入。
        如果没有，则使用嵌入模型对查询进行编码，并存储编码后的嵌入结果。

        参数:
            queries (List[str] | List[QuerySolution]): 查询字符串列表或 QuerySolution 对象列表。
        """

        all_query_strings = []
        # 遍历所有查询，收集尚未编码的查询字符串
        for query in queries:
            if isinstance(query, QuerySolution) and query.question not in self.query_to_embedding['passage']:
                # 如果是 QuerySolution 对象且未编码，则添加其 question 字段
                all_query_strings.append(query.question)
            elif query not in self.query_to_embedding['passage']:
                # 如果是字符串且未编码，则直接添加
                all_query_strings.append(query)

        if len(all_query_strings) > 0:
            # 对所有未编码的查询进行编码，获取 query_to_passage 的嵌入
            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_passage.")
            query_embeddings_for_passage = self.embedding_model.batch_encode(
                all_query_strings,
                instruction=get_query_instruction('query_to_passage'),
                norm=True
            )
            # 存储每个查询的 passage 嵌入
            for query, embedding in zip(all_query_strings, query_embeddings_for_passage):
                self.query_to_embedding['passage'][query] = embedding


    def _extract_query_entities(self, query: str) -> List[str]:
        """
        使用已有的 OpenIE NER 提取查询中的实体，并做简单缓存。
        """
        cached = self._query_entity_cache.get(query)
        if cached is not None:
            return cached

        entities: List[str] = []
        try:
            ner_output = self.openie.ner(chunk_key=compute_mdhash_id(query, "query-"), passage=query)
            if ner_output and ner_output.unique_entities:
                entities = [e.strip() for e in ner_output.unique_entities if isinstance(e, str) and e.strip()]
                # 去重并保序
                entities = list(dict.fromkeys(entities))
        except Exception as e:
            logger.warning(f"Query NER failed, fallback to no entities: {e}")
            entities = []

        self._query_entity_cache[query] = entities
        return entities


    def _get_entity_seed_scores(self, query: str) -> Tuple[List[Tuple[int, float]], Dict[str, Any]]:
        """
        计算查询实体与库内实体的相似度，返回命中的实体索引及分数。
        """
        cached = self._query_entity_score_cache.get(query)
        if cached is not None:
            return cached

        metadata: Dict[str, Any] = {"extracted_entities": [], "selected_entities": []}

        entity_embeddings = getattr(self, "entity_embeddings", None)
        if entity_embeddings is None:
            result = ([], metadata)
            self._query_entity_score_cache[query] = result
            return result
        if isinstance(entity_embeddings, np.ndarray):
            if entity_embeddings.size == 0:
                result = ([], metadata)
                self._query_entity_score_cache[query] = result
                return result
        elif len(entity_embeddings) == 0:
            result = ([], metadata)
            self._query_entity_score_cache[query] = result
            return result

        entities = self._extract_query_entities(query)
        metadata["extracted_entities"] = entities
        if not entities:
            result = ([], metadata)
            self._query_entity_score_cache[query] = result
            return result

        try:
            entity_vecs = self._query_entity_emb_cache.get(query)
            if entity_vecs is None:
                # Try to use cached text embeddings; encode missing ones in a batch
                missing_texts = [e for e in entities if e not in self._entity_text_emb_cache]
                if missing_texts:
                    try:
                        vecs = self.embedding_model.batch_encode(
                            missing_texts,
                            instruction=get_query_instruction('query_to_node'),
                            norm=True
                        )
                        for text, vec in zip(missing_texts, vecs):
                            self._entity_text_emb_cache[text] = np.array(vec)
                    except Exception as e:
                        logger.debug(f"Entity encoding (fallback) failed: {e}")

                vec_list = []
                for e in entities:
                    vec = self._entity_text_emb_cache.get(e)
                    if vec is not None:
                        vec_list.append(vec)

                if vec_list:
                    entity_vecs = np.array(vec_list)
                    self._query_entity_emb_cache[query] = entity_vecs
                else:
                    entity_vecs = np.array([])
        except Exception as e:
            logger.warning(f"Encoding query entities failed, skip entity seeds: {e}")
            result = ([], metadata)
            self._query_entity_score_cache[query] = result
            return result

        entity_vecs = np.array(entity_vecs)
        if entity_vecs.ndim == 1:
            entity_vecs = entity_vecs.reshape(1, -1)

        try:
            sim_matrix = np.dot(self.entity_embeddings, entity_vecs.T)
        except Exception as e:
            logger.warning(f"Entity similarity computation failed, skip entity seeds: {e}")
            result = ([], metadata)
            self._query_entity_score_cache[query] = result
            return result

        if sim_matrix.size == 0:
            result = ([], metadata)
            self._query_entity_score_cache[query] = result
            return result

        sim_scores = np.max(sim_matrix, axis=1)
        threshold = getattr(self.global_config, "entity_sim_threshold", 0.0) or 0.0
        top_k = getattr(self.global_config, "entity_top_k", None)
        if top_k is None or top_k <= 0:
            top_k = len(self.entity_node_keys)

        ranked = np.argsort(sim_scores)[::-1]
        selected_idx: List[int] = []
        for idx in ranked:
            if sim_scores[idx] < threshold:
                continue
            selected_idx.append(idx)
            if len(selected_idx) >= top_k:
                break
        if not selected_idx and sim_scores.size > 0:
            # 保底保留最高分实体，避免小样本场景直接空
            selected_idx = [ranked[0]]

        selected: List[Tuple[int, float]] = [(idx, float(sim_scores[idx])) for idx in selected_idx]

        # Collect selected entities for logging/debugging
        for idx, score in selected:
            try:
                ent_text = self.entity_embedding_store.get_row(self.entity_node_keys[idx])["content"]
            except Exception:
                ent_text = None
            metadata["selected_entities"].append({"entity": ent_text, "score": score})

        result = (selected, metadata)
        self._query_entity_score_cache[query] = result
        return result


    async def prefetch_query_entities_async(self, queries: List[str]):
        """
        异步并发预取查询的实体抽取结果（仅 NER），并缓存。
        并发度直接使用 OpenIE 内部信号量（max_concurrency）。
        """
        if not queries:
            return

        sem = self.openie._get_semaphore()
        # Per-task timeout: only counts actual execution time (after acquiring semaphore)
        task_timeout = 60.0

        async def _prefetch_single(query_text: str):
            chunk_key = compute_mdhash_id(query_text, "query-")
            try:
                async with sem:
                    res = await asyncio.wait_for(
                        self.openie.ner_async(chunk_key, query_text, False),
                        timeout=task_timeout
                    )
                entities = []
                if res and getattr(res, "unique_entities", None):
                    entities = [e.strip() for e in res.unique_entities if isinstance(e, str) and e.strip()]
                    entities = list(dict.fromkeys(entities))
                self._query_entity_cache[query_text] = entities
            except asyncio.TimeoutError:
                logger.warning(f"Entity extraction prefetch timed out for query '{query_text[:80]}'")
                self._query_entity_cache[query_text] = []
            except Exception as e:
                logger.debug(f"Entity extraction prefetch failed for query '{query_text[:80]}': {e}")
                self._query_entity_cache[query_text] = []

        tasks = [asyncio.create_task(_prefetch_single(q_text)) for q_text in queries]

        progress_iter = tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Prefetch query entities",
            leave=False,
            disable=False if not self.openie._disable_progress() else True,
            mininterval=0.5
        )
        for fut in progress_iter:
            await fut

    async def prefetch_query_entity_embeddings_async(self, queries: List[str]):
        """
        异步并发预计算查询实体的 embedding（在实体抽取已完成的基础上）。
        使用跨查询去重的批量编码，减少调用次数。
        """
        if not queries:
            return

        # 收集每个查询的实体列表
        entities_by_query: Dict[str, List[str]] = {}
        for q_text in queries:
            ents = self._query_entity_cache.get(q_text)
            if ents is None:
                ents = self._extract_query_entities(q_text)
            entities_by_query[q_text] = ents or []

        # 跨查询去重实体文本
        all_entities = [e for ents in entities_by_query.values() for e in ents]
        unique_entities = list(dict.fromkeys(all_entities))

        # 找出需要编码的实体（未在全局缓存中）
        to_encode = [e for e in unique_entities if e not in self._entity_text_emb_cache]
        if to_encode:
            batch_size = max(1, getattr(self.global_config, "embedding_batch_size", 16))
            batches = [to_encode[i:i + batch_size] for i in range(0, len(to_encode), batch_size)]
            progress_iter = tqdm(
                batches,
                total=len(batches),
                desc="Prefetch entity embeddings",
                leave=False,
                disable=False,
                mininterval=0.5
            )
            for batch in progress_iter:
                try:
                    vecs = self.embedding_model.batch_encode(
                        batch,
                        instruction=get_query_instruction('query_to_node'),
                        norm=True
                    )
                    for text, vec in zip(batch, vecs):
                        self._entity_text_emb_cache[text] = np.array(vec)
                except Exception as e:
                    logger.debug(f"Entity embedding batch failed: {e}")

        # 按查询组装并缓存
        for q_text, ents in entities_by_query.items():
            if q_text in self._query_entity_emb_cache:
                continue
            vec_list = []
            for e in ents:
                vec = self._entity_text_emb_cache.get(e)
                if vec is not None:
                    vec_list.append(vec)
            if vec_list:
                self._query_entity_emb_cache[q_text] = np.array(vec_list)

    def prefetch_query_entities(self, queries: List[str]):
        """
        同步版本：在独立的 event loop 中运行实体预取，与 NER 阶段保持一致的调用模式。
        """
        return self._run_async(self.prefetch_query_entities_async(queries))

    def prefetch_query_entity_embeddings(self, queries: List[str]):
        """
        同步版本：在独立的 event loop 中运行实体 embedding 预取。
        """
        return self._run_async(self.prefetch_query_entity_embeddings_async(queries))

    def dense_passage_retrieval(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Conduct dense passage retrieval to find relevant documents for a query.

        This function processes a given query using a pre-trained embedding model
        to generate query embeddings. The similarity scores between the query
        embedding and passage embeddings are computed using dot product, followed
        by score normalization. Finally, the function ranks the documents based
        on their similarity scores and returns the ranked document identifiers
        and their scores.

        Parameters
        ----------
        query : str
            The input query for which relevant passages should be retrieved.

        Returns
        -------
        tuple : Tuple[np.ndarray, np.ndarray]
            A tuple containing two elements:
            - A list of sorted document identifiers based on their relevance scores.
            - A numpy array of the normalized similarity scores for the corresponding
              documents.
        """
        query_embedding = self.query_to_embedding['passage'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_passage'),
                                                                norm=True)
        query_doc_scores = np.dot(self.fragment_embeddings, query_embedding.T)
        query_doc_scores = np.squeeze(query_doc_scores) if query_doc_scores.ndim == 2 else query_doc_scores
        query_doc_scores = min_max_normalize(query_doc_scores)

        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]
        return sorted_doc_ids, sorted_doc_scores


    def graph_search_with_fact_entities(self, query: str,
                                        passage_node_weight: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes document scores using Personalized PageRank (PPR) and dense retrieval.

        This method uses DPR to weight fragment nodes, then runs PPR on the graph structure
        (which includes entity-entity edges from triples) to propagate weights and find
        relevant fragments through multi-hop reasoning.

        Parameters:
            query (str): The input query string
            passage_node_weight (float): Weight to scale passage scores in the graph

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Document IDs sorted by relevance scores
                - PPR scores for the sorted documents
        """

        # Fragment-centric architecture with optional entity seeding
        entity_weights = np.zeros(len(self.graph.vs['name']))
        passage_weights = np.zeros(len(self.graph.vs['name']))

        # Get passage scores from dense retrieval model
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)

        # Assign DPR scores to fragment nodes
        for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            fragment_node_key = self.fragment_node_keys[dpr_sorted_doc_id]
            fragment_dpr_score = normalized_dpr_sorted_scores[i]
            fragment_node_id = self.node_name_to_vertex_idx[fragment_node_key]
            passage_weights[fragment_node_id] = fragment_dpr_score * passage_node_weight

        # Entity seeding: extract query entities -> embed -> similarity against entity store
        entity_seed_pairs, _ = self._get_entity_seed_scores(query)
        for entity_idx, score in entity_seed_pairs:
            if 0 <= entity_idx < len(self.entity_node_idxs):
                node_id = self.entity_node_idxs[entity_idx]
                entity_weights[node_id] = score

        # Combining entity and passage scores into one array for PPR
        node_weights = passage_weights
        if entity_seed_pairs:
            node_weights = node_weights + (entity_weights * self.global_config.entity_node_weight)

        assert sum(node_weights) > 0, f'ERROR: No seeds assigned. DPR/NER may have failed for query: {query[:100]}'

        # Running PPR algorithm based on the passage weights
        # PPR will propagate weights through the graph structure (including entity-entity edges)
        ppr_start = time.time()
        ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(node_weights, damping=self.global_config.damping)
        ppr_end = time.time()

        with self._metrics_lock:
            self.ppr_time += (ppr_end - ppr_start)

        assert len(ppr_sorted_doc_ids) == len(
            self.fragment_node_idxs), f"Doc prob length {len(ppr_sorted_doc_ids)} != corpus length {len(self.fragment_node_idxs)}"

        return ppr_sorted_doc_ids, ppr_sorted_doc_scores



    def run_ppr(self,
                reset_prob: np.ndarray,
                damping: float =0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs Personalized PageRank (PPR) on a graph and computes relevance scores for
        nodes corresponding to document passages. The method utilizes a damping
        factor for teleportation during rank computation and can take a reset
        probability array to influence the starting state of the computation.

        Parameters:
            reset_prob (np.ndarray): A 1-dimensional array specifying the reset
                probability distribution for each node. The array must have a size
                equal to the number of nodes in the graph. NaNs or negative values
                within the array are replaced with zeros.
            damping (float): A scalar specifying the damping factor for the
                computation. Defaults to 0.3 if not provided or set to `None`.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays. The
                first array represents the sorted node IDs of document passages based
                on their relevance scores in descending order. The second array
                contains the corresponding relevance scores of each document passage
                in the same order.
        """

        if damping is None: damping = 0.3 # for potential compatibility
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)

        # THREAD SAFETY: Protect igraph operations with lock
        # python-igraph is NOT thread-safe and concurrent calls can cause crashes
        with self._graph_lock:
            pagerank_scores = self.graph.personalized_pagerank(
                vertices=range(len(self.node_name_to_vertex_idx)),
                damping=damping,
                directed=False,
                weights='weight',
                reset=reset_prob,
                implementation='prpack'
            )

        doc_scores = np.array([pagerank_scores[idx] for idx in self.fragment_node_idxs])
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]

        return sorted_doc_ids, sorted_doc_scores

import os
import argparse
import json
import logging
import shutil
from typing import Dict, List
from pathlib import Path
from datetime import datetime

import asyncio
from tqdm import tqdm

# Load environment variables from centralized LLM config
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs" / "atomicrag"
DATASET_DIR = REPO_ROOT / "dataset"  # Updated to use new dataset directory
WORKSPACE_ROOT = REPO_ROOT / "workspaces" / "atomicrag"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "atomicrag"
RESULTS_ROOT = DEFAULT_RESULTS_ROOT  # can be overridden via CLI
LOGS_DIR = REPO_ROOT / "logs" / "atomicrag"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

CONFIG_JSON_PATH = CONFIG_DIR / "config.json"

# Import AtomicRAG components early
from atomicrag.utils.config_utils import BaseConfig
from atomicrag.atomicrag import AtomicRAG

def apply_json_config_overrides(cfg: BaseConfig) -> BaseConfig:
    if not CONFIG_JSON_PATH.exists():
        return cfg

    try:
        with open(CONFIG_JSON_PATH, "r", encoding="utf-8") as f:
            config_data = json.load(f)
    except Exception as exc:
        logging.warning(f"Failed to load config overrides from {CONFIG_JSON_PATH}: {exc}")
        return cfg

    openai_cfg = config_data.get("models", {}).get("openai", {})
    concurrency_value = openai_cfg.get("max_concurrency") or openai_cfg.get("api_max_workers")
    if concurrency_value is not None:
        cfg.max_concurrency = concurrency_value

    return cfg

# Apply environment variables from config.json
if CONFIG_JSON_PATH.exists():
    try:
        with open(CONFIG_JSON_PATH, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        env_vars = config_data.get("environment", {})
        for key, value in env_vars.items():
            os.environ.setdefault(key, str(value))
    except Exception as exc:
        logging.warning(f"Failed to apply environment variables from {CONFIG_JSON_PATH}: {exc}")

# Set CUDA device (allow override via config)
# Default to env var if already set by config.json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "atomicrag_processing.log")
    ]
)

# Suppress HTTP request logs from httpx and openai
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

def group_questions_by_source(question_list: List[dict]) -> Dict[str, List[dict]]:
    """Group questions by their source"""
    grouped_questions = {}
    for question in question_list:
        source = question.get("source")
        if source not in grouped_questions:
            grouped_questions[source] = []
        grouped_questions[source].append(question)
    return grouped_questions

async def process_corpus(
    corpus_name: str,
    chunks: List[str],
    base_dir: str,
    model_name: str,
    embed_model_path: str,
    llm_base_url: str,
    questions: Dict[str, List[dict]],
    sample: int,
    use_cache: bool = False,
    max_concurrency: int | None = None,
    qa_prompt_template: str | None = None,
    enable_fragment_filter: bool = True,
    enable_query_decomposition: bool = True,
    enable_ppr: bool = True
):
    """Process a single corpus: index it and answer its questions"""
    logging.info(f"📚 Processing corpus: {corpus_name}")

    try:
        # If user opts out of cache, wipe any previous workspace to force a full rebuild
        cache_path = Path(base_dir) / corpus_name
        if not use_cache and cache_path.exists():
            logging.info(f"🧹 Clearing existing workspace for fresh rebuild: {cache_path}")
            shutil.rmtree(cache_path)

        # Prepare output directory
        output_dir = RESULTS_ROOT / corpus_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"predictions_{corpus_name}.json"

        # Skip if predictions already exist and using cache mode
        if use_cache and output_path.exists():
            logging.info(f"⏭️  Skipping {corpus_name}: predictions file already exists at {output_path}")
            return

        # Format chunks as documents (chunks are already provided from JSON)
        logging.info(f"📦 Using {len(chunks)} pre-chunked documents")
        docs = [f'{idx}:{chunk}' for idx, chunk in enumerate(chunks)]

        # Get questions for this corpus
        corpus_questions = questions.get(corpus_name, [])
        if not corpus_questions:
            logging.warning(f"⚠️ No questions found for corpus: {corpus_name}")
            raise ValueError(f"No questions found for corpus '{corpus_name}'")

        # Sample questions if requested
        if sample and sample < len(corpus_questions):
            corpus_questions = corpus_questions[:sample]

        logging.info(f"🔍 Found {len(corpus_questions)} questions for {corpus_name}")

        # Prepare queries
        all_queries = [q["question"] for q in corpus_questions]

        # Configure AtomicRAG
        def build_config(force_rebuild: bool) -> BaseConfig:
            cfg = BaseConfig(
                save_dir=str(Path(base_dir) / corpus_name),  # 索引与结果缓存目录
                llm_base_url=llm_base_url,  # LLM 服务的基础地址
                llm_name=model_name,  # 调用的 LLM 模型名称
                embedding_model_name=embed_model_path,  # 用于向量化的embedding模型
                force_index_from_scratch=force_rebuild,  # 是否强制重建向量索引
                force_openie_from_scratch=force_rebuild,  # 是否强制重新抽取事实（OpenIE）
                retrieval_top_k=25,  # Paper setting: retrieved candidate atoms per query
                qa_top_k=25,  # Keep QA context budget aligned with retrieval_top_k
                passage_node_weight=0.1,  # Paper setting for atom/passage seeds
                damping=0.3,  # Paper setting for PPR restart/damping coefficient
                embedding_batch_size=128,  # 批量生成向量的并行大小
                max_new_tokens=None,  # 回答时允许生成的新token上限（None表示使用默认）
                enable_fragment_filter=enable_fragment_filter,
                enable_query_decomposition=enable_query_decomposition,
                enable_ppr=enable_ppr,
                max_retry_attempts=5   # 失败重试次数（默认5次）
            )

            if max_concurrency is not None:
                cfg.max_concurrency = max_concurrency

            if qa_prompt_template is not None:
                cfg.qa_prompt_template = qa_prompt_template

            cfg = apply_json_config_overrides(cfg)
            # CLI toggles take precedence over JSON overrides
            cfg.enable_fragment_filter = enable_fragment_filter
            cfg.enable_query_decomposition = enable_query_decomposition
            cfg.enable_ppr = enable_ppr
            return cfg

        force_rebuild = not use_cache
        config = build_config(force_rebuild)

        cache_mode = "🔄 Using cached index/extraction (if exists)" if use_cache else "🔨 Rebuilding from scratch"
        logging.info(f"{cache_mode}")
        logging.info(f"✅ Using OpenAI mode: {model_name} at {llm_base_url}")

        # Initialize AtomicRAG with fallback in case cache is inconsistent
        def initialize_system(cfg: BaseConfig) -> AtomicRAG:
            return AtomicRAG(global_config=cfg)

        rag_system = initialize_system(config)

        try:
            rag_system.index(docs)
            logging.info(f"✅ Indexed corpus: {corpus_name}")
        except AssertionError as err:
            if use_cache:
                logging.warning(f"⚠️ Cache inconsistency detected for '{corpus_name}'. Purging workspace and rebuilding. Details: {err}")
                cache_path = Path(base_dir) / corpus_name
                if cache_path.exists():
                    shutil.rmtree(cache_path)
                # Rebuild from scratch
                config = build_config(force_rebuild=True)
                rag_system = initialize_system(config)
                rag_system.index(docs)
                logging.info(f"✅ Indexed corpus: {corpus_name} (after cache rebuild)")
            else:
                raise

        # Process questions
        results = []

        rag_outputs = await rag_system.rag_qa_async(queries=all_queries)
        queries_solutions = rag_outputs[0]
        solutions = [query.to_dict() for query in queries_solutions]

        for question in corpus_questions:
            solution = next((sol for sol in solutions if sol['question'] == question['question']), None)
            if solution:
                results.append({
                    "id": question["id"],
                    "question": question["question"],
                    "source": corpus_name,
                    "context": solution.get("docs", ""),
                    "evidence": question.get("evidence", ""),
                    "question_type": question.get("question_type", ""),
                    "generated_answer": solution.get("answer", ""),
                    "ground_truth": question.get("answer", ""),
                    "decomposition_metadata": solution.get("decomposition_metadata")
                })

        # Save results
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logging.info(f"💾 Saved {len(results)} predictions to: {output_path}")

        # Save statistics
        rag_system.stats.corpus_name = corpus_name
        rag_system.stats.num_questions = len(corpus_questions)
        statistics = rag_system.get_statistics()

        # Generate statistics filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_filename = f"statistics_{corpus_name}_{timestamp}.json"
        stats_path = output_dir / stats_filename

        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)

        logging.info(f"📊 Saved statistics to: {stats_path}")

    except Exception:
        logging.exception(f"❌ Failed to process corpus '{corpus_name}'")
        raise

def main():
    parser = argparse.ArgumentParser(description="AtomicRAG: Process Corpora and Answer Questions")

    # Core arguments
    parser.add_argument("--subset", required=True,
                        help="Subset to process (medical, novel, 2wikimultihop, hotpotqa, or musique)")
    parser.add_argument("--base_dir", default=str(WORKSPACE_ROOT),
                        help="Base working directory for AtomicRAG")

    # Model configuration
    parser.add_argument("--model_name", default="gpt-4o-mini",
                        help="LLM model identifier")
    parser.add_argument("--embed_model_path", default="BAAI/bge-large-en-v1.5",
                        help="Path to embedding model directory")
    parser.add_argument("--sample", type=int, default=None,
                        help="Number of questions to sample per corpus")

    # API configuration
    parser.add_argument("--llm_base_url", default=os.getenv("LLM_BASE_URL"),
                        help="Base URL for LLM API")
    parser.add_argument("--llm_api_key", default=None,
                        help="API key for LLM service (fallback to OPENAI_API_KEY environment variable)")
    parser.add_argument("--use_cache", choices=["true", "false"], default="false",
                        help="Whether to use cached index and extraction (true) or rebuild from scratch (false)")
    parser.add_argument("--concurrency", type=int, default=200,
                        help="Maximum concurrent requests for API calls")
    parser.add_argument("--results_dir", default=None,
                        help="Directory to save results; defaults to results/atomicrag")
    parser.add_argument("--qa_prompt_template", default=None,
                        help="Override QA prompt template (e.g., 'precise' for rag_qa_precise, 'abstract' for rag_qa_abstract)")
    parser.add_argument("--disable_fragment_filter", action="store_true",
                        help="Disable fragment/question filtering stage for ablation")
    parser.add_argument("--disable_query_decomposition", action="store_true",
                        help="Disable query decomposition module for ablation runs")
    parser.add_argument("--disable_ppr", action="store_true",
                        help="Skip graph-based PPR and use DPR-only retrieval")

    args = parser.parse_args()

    # Override results root if provided
    global RESULTS_ROOT
    if args.results_dir:
        RESULTS_ROOT = Path(args.results_dir)
    else:
        RESULTS_ROOT = DEFAULT_RESULTS_ROOT
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    logging.info(f"🚀 Starting AtomicRAG processing for subset: {args.subset}")

    # Scan for sub-datasets in the subset directory
    subset_dir = DATASET_DIR / args.subset
    if not subset_dir.exists():
        logging.error(f"❌ Subset directory not found: {subset_dir}")
        return

    # Handle API key security
    api_key = args.llm_api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        logging.warning("⚠️ No API key provided! Requests may fail.")

    # Create workspace directory
    os.makedirs(args.base_dir, exist_ok=True)

    # Scan for sub-datasets (directories containing chunks.json and questions.json)
    sub_datasets = []
    for sub_dir in sorted(subset_dir.iterdir()):
        if not sub_dir.is_dir():
            continue

        chunks_path = sub_dir / "chunks.json"
        questions_path = sub_dir / "questions.json"

        if chunks_path.exists() and questions_path.exists():
            sub_datasets.append({
                "name": sub_dir.name,
                "chunks_path": chunks_path,
                "questions_path": questions_path
            })

    if not sub_datasets:
        logging.error(f"❌ No valid sub-datasets found in {subset_dir}")
        return

    logging.info(f"📁 Found {len(sub_datasets)} sub-datasets in {subset_dir}")

    # Load data for all sub-datasets
    corpus_data = []
    all_questions = []

    for sub_dataset in sub_datasets:
        name = sub_dataset["name"]

        try:
            # Load chunks
            with open(sub_dataset["chunks_path"], 'r', encoding='utf-8') as f:
                chunks = json.load(f)

            # Load questions
            with open(sub_dataset["questions_path"], 'r', encoding='utf-8') as f:
                questions = json.load(f)

            # Ensure all questions have corpus name as source (overwrite existing source)
            for q in questions:
                q["source"] = name

            corpus_data.append({
                "corpus_name": name,
                "chunks": chunks
            })

            all_questions.extend(questions)

            logging.info(f"  ✅ {name}: {len(chunks)} chunks, {len(questions)} questions")

        except Exception as e:
            logging.error(f"  ❌ Failed to load {name}: {e}")
            continue

    # Group questions by source
    grouped_questions = group_questions_by_source(all_questions)
    logging.info(f"✅ Total: {len(corpus_data)} corpora, {len(all_questions)} questions")
    logging.info(f"📊 Grouped questions by source: {dict((k, len(v)) for k, v in grouped_questions.items())}")

    # Sample corpus data if requested
    if args.sample:
        corpus_data = corpus_data[:args.sample] if args.sample < len(corpus_data) else corpus_data
        logging.info(f"📊 Sampling {len(corpus_data)} corpora")

    # Convert use_cache string to boolean
    use_cache = args.use_cache.lower() == "true"

    # Process each corpus in the subset
    async def _run_all():
        # 限制同时处理的语料库数量为1
        max_concurrent_corpus = 1
        semaphore = asyncio.Semaphore(max_concurrent_corpus)

        async def process_with_limit(item):
            async with semaphore:
                return await process_corpus(
                    corpus_name=item["corpus_name"],
                    chunks=item["chunks"],
                    base_dir=args.base_dir,
                    model_name=args.model_name,
                    embed_model_path=args.embed_model_path,
                    llm_base_url=args.llm_base_url,
                    questions=grouped_questions,
                    sample=args.sample,
                    use_cache=use_cache,
                    max_concurrency=args.concurrency,
                    qa_prompt_template=args.qa_prompt_template,
                    enable_fragment_filter=not args.disable_fragment_filter,
                    enable_query_decomposition=not args.disable_query_decomposition,
                    enable_ppr=not args.disable_ppr
                )

        tasks = [process_with_limit(item) for item in corpus_data]
        await asyncio.gather(*tasks)

    asyncio.run(_run_all())

if __name__ == "__main__":
    main()

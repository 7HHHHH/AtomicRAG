import asyncio
import argparse
import json
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dotenv import load_dotenv
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from Evaluation.metrics import compute_answer_correctness, SkipSampleError

# Load shared LLM configuration
LLM_ENV_PATH = Path(__file__).resolve().parents[1] / "configs" / "atomicrag" / "llm.env"
load_dotenv(LLM_ENV_PATH)
CONFIG_JSON_PATH = Path(__file__).resolve().parents[1] / "configs" / "atomicrag" / "config.json"

SEED = 42


def _load_default_concurrency() -> int:
    """Load default evaluation concurrency from config.json if present."""
    try:
        with open(CONFIG_JSON_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # Prefer evaluation block; fallback to openai model setting; then hard default
        eval_cfg = cfg.get("evaluation", {})
        if isinstance(eval_cfg, dict) and eval_cfg.get("max_concurrency") is not None:
            return int(eval_cfg["max_concurrency"])
        model_cfg = cfg.get("models", {}).get("openai", {})
        if isinstance(model_cfg, dict) and model_cfg.get("max_concurrency") is not None:
            return int(model_cfg["max_concurrency"])
    except Exception:
        pass
    return 20  # hard default

async def evaluate_dataset(
    dataset: Dataset,
    metrics: List[str],
    llm: Any,
    embeddings: Embeddings,
    max_concurrent: int = 100,  # Limit concurrent evaluations
    detailed_output: bool = False
) -> Dict[str, Any]:
    """Evaluate the metric scores on the entire dataset."""
    results = {metric: [] for metric in metrics}
    detailed_results = [] if detailed_output else None

    ids = dataset["id"]
    questions = dataset["question"]
    answers = dataset["answer"]
    contexts_list = dataset["contexts"]
    ground_truths = dataset["ground_truth"]

    total_samples = len(questions)
    print(f"\nStarting evaluation of {total_samples} samples...")

    worker_limit = max(1, max_concurrent)
    sample_results = []
    completed = 0

    async def _run_single(idx: int, max_retries: int = 3):
        last_error = None
        for attempt in range(max_retries):
            try:
                sample_metrics = await evaluate_sample(
                    question=questions[idx],
                    answer=answers[idx],
                    contexts=contexts_list[idx],
                    ground_truth=ground_truths[idx],
                    metrics=metrics,
                    llm=llm,
                    embeddings=embeddings
                )
                if detailed_output:
                    return True, {
                        "id": ids[idx],
                        "question": questions[idx],
                        "ground_truth": ground_truths[idx],
                        "generated_answer": answers[idx],
                        "contexts": contexts_list[idx],
                        "metrics": sample_metrics
                    }
                return True, sample_metrics
            except SkipSampleError as skip_err:
                # Skip this sample - don't retry, just return None to indicate skip
                print(f"⏭️  Sample {idx} skipped: {skip_err}")
                return False, None
            except Exception as exc:
                last_error = exc
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue

        # All retries failed, return 0 scores for all metrics
        print(f"❌ Sample {idx} failed after {max_retries} attempts: {last_error}, scoring as 0")
        zero_metrics = {metric: 0.0 for metric in metrics}
        if detailed_output:
            return True, {
                "id": ids[idx],
                "question": questions[idx],
                "ground_truth": ground_truths[idx],
                "generated_answer": answers[idx],
                "contexts": contexts_list[idx],
                "metrics": zero_metrics
            }
        return True, zero_metrics

    in_flight = set()
    next_idx = 0

    def _start_task(i: int):
        return asyncio.create_task(_run_single(i))

    while next_idx < total_samples and len(in_flight) < worker_limit:
        in_flight.add(_start_task(next_idx))
        next_idx += 1

    while in_flight:
        done, in_flight = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            success, payload = task.result()

            # Handle skipped samples - just continue without recording
            if not success:
                while next_idx < total_samples and len(in_flight) < worker_limit:
                    in_flight.add(_start_task(next_idx))
                    next_idx += 1
                continue

            completed += 1

            if detailed_output and detailed_results is not None:
                detailed_results.append(payload)
                metrics_dict = payload.get("metrics") if isinstance(payload, dict) else None
                if isinstance(metrics_dict, dict):
                    for metric, score in metrics_dict.items():
                        if isinstance(score, (int, float)) and not np.isnan(score):
                            results[metric].append(score)
            else:
                sample_results.append(payload)
                if isinstance(payload, dict):
                    for metric, score in payload.items():
                        if isinstance(score, (int, float)) and not np.isnan(score):
                            results[metric].append(score)
            print(f"✅ Completed {completed}/{total_samples} - {(completed/total_samples)*100:.1f}%")

            while next_idx < total_samples and len(in_flight) < worker_limit:
                in_flight.add(_start_task(next_idx))
                next_idx += 1

    avg_results = {metric: np.nanmean(scores) for metric, scores in results.items()}

    if detailed_output:
        return {
            "average_scores": avg_results,
            "detailed": detailed_results
        }
    else:
        return avg_results

async def evaluate_sample(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str,
    metrics: List[str],
    llm: Any,
    embeddings: Embeddings
) -> Dict[str, float]:
    """Evaluate the metric scores for a single sample."""
    results = {}

    # Only compute answer_correctness
    if "answer_correctness" in metrics:
        results["answer_correctness"] = await compute_answer_correctness(
            question, answer, ground_truth, llm, embeddings
        )

    return results

async def main(args: argparse.Namespace):
    """Main evaluation function that accepts command-line arguments."""
    # Check if the API key is set
    if not os.getenv("LLM_API_KEY"):
        raise ValueError("LLM_API_KEY environment variable is not set")

    # Initialize the model
    # Wrap API key in SecretStr to satisfy type hints
    from pydantic import SecretStr
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise ValueError("LLM_API_KEY environment variable is not set")
    llm = ChatOpenAI(
        model=args.model,
        base_url=args.base_url,
        api_key=SecretStr(api_key),
        temperature=0.0,
        max_retries=3,
        timeout=30,
        model_kwargs={
            "top_p": 1,
            "seed": SEED,
            "presence_penalty": 0,
            "frequency_penalty": 0
        }
    )

    # Initialize the embedding model
    # Set environment variable to prevent network access
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    embedding = HuggingFaceBgeEmbeddings(model_name=args.embedding_model)

    # Load evaluation data
    print(f"Loading evaluation data from {args.data_file}...")
    with open(args.data_file, 'r') as f:
        file_data = json.load(f)  # Now a list of question items

    # Map multi-hop QA dataset question types to "Complex Reasoning"
    multihop_types = {'compositional', 'comparison', 'bridge_comparison', 'inference', 'bridge'}
    for item in file_data:
        if item.get('question_type') in multihop_types:
            item['question_type'] = 'Complex Reasoning'

    # Define the evaluation metrics for each question type
    # Only using answer_correctness for all types
    metric_config = {
        'Fact Retrieval': ["answer_correctness"],
        'Complex Reasoning': ["answer_correctness"],
        'Contextual Summarize': ["answer_correctness"],
        'Creative Generation': ["answer_correctness"]
    }

    # Group data by question type
    grouped_data = {}
    for item in file_data:
        q_type = item.get("question_type", "Uncategorized")
        if q_type not in grouped_data:
            grouped_data[q_type] = []
        grouped_data[q_type].append(item)

    all_results = {}

    # Evaluate each found question type (only those in metric_config)
    for question_type in list(grouped_data.keys()):
        # Skip types not defined in metric_config
        if question_type not in metric_config:
            print(f"Skipping undefined question type: {question_type}")
            continue

        print(f"\n{'='*50}")
        print(f"Evaluating question type: {question_type}")
        print(f"{'='*50}")

        # Prepare data from grouped items
        group_items = grouped_data[question_type]
        ids = [item['id'] for item in group_items]
        questions = [item['question'] for item in group_items]
        ground_truths = [item['ground_truth'] for item in group_items]
        answers = [item['generated_answer'] for item in group_items]
        contexts = [item['context'] for item in group_items]

        # Create dataset
        data = {
            "id": ids,
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        dataset = Dataset.from_dict(data)

        # If sample
        if args.num_samples:
            dataset = dataset.select([i for i in list(range(args.num_samples))])

        # Perform evaluation
        results = await evaluate_dataset(
            dataset=dataset,
            metrics=metric_config[question_type],
            llm=llm,
            embeddings=embedding,
            max_concurrent=args.concurrency,
            detailed_output=args.detailed_output
        )

        all_results[question_type] = results
        print(f"\nResults for {question_type}:")
        if args.detailed_output:
            for metric, score in results["average_scores"].items():
                print(f"  {metric}: {score:.4f}")
        else:
            for metric, score in results.items():
                print(f"  {metric}: {score:.4f}")

    # Save final results
    if args.output_file:
        print(f"\nSaving results to {args.output_file}...")
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

    print('\nEvaluation complete.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate RAG performance using answer_correctness metric",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use for evaluation"
    )

    parser.add_argument(
        "--base_url",
        type=str,
        default=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        help="Base URL for the OpenAI API"
    )

    parser.add_argument(
        "--embedding_model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="HuggingFace model for embeddings"
    )

    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to JSON file containing evaluation data"
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_results.json",
        help="Path to save evaluation results"
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to use for evaluation"
    )

    parser.add_argument(
        "--detailed_output",
        action="store_true",
        help="Whether to include detailed output"
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=_load_default_concurrency(),
        help="Maximum concurrent evaluations (default reads configs/atomicrag/config.json:evaluation.max_concurrency)"
    )

    args = parser.parse_args()

    asyncio.run(main(args))

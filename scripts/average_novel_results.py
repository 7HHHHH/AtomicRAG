#!/usr/bin/env python3
"""
Average Novel Dataset Evaluation Results
计算所有Novel子数据集的评估结果平均值
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("average_novel_results.log")
    ]
)


def load_eval_results(eval_file: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file"""
    try:
        with open(eval_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"Evaluation file not found: {eval_file}")
        return None
    except json.JSONDecodeError:
        logging.warning(f"Invalid JSON in evaluation file: {eval_file}")
        return None


def aggregate_results(results_dir: str) -> Dict[str, List[Dict[str, float]]]:
    """
    Aggregate all evaluation results from Novel subdatasets

    Returns:
        Dictionary mapping question_type -> list of metric dictionaries
    """
    aggregated = defaultdict(lambda: defaultdict(list))
    found_results = 0

    # Find all Novel-* directories
    for item in Path(results_dir).iterdir():
        if item.is_dir() and item.name.startswith("Novel-"):
            eval_file = item / f"eval_generation_{item.name}.json"

            if eval_file.exists():
                results = load_eval_results(str(eval_file))

                if results:
                    found_results += 1
                    logging.info(f"✅ Loaded {item.name}")

                    # Aggregate metrics by question type
                    for question_type, metrics in results.items():
                        for metric_name, metric_value in metrics.items():
                            aggregated[question_type][metric_name].append(metric_value)
                else:
                    logging.warning(f"⚠️  Failed to load {item.name}")
            else:
                logging.debug(f"No eval file found for {item.name}")

    if found_results == 0:
        logging.error("❌ No evaluation results found!")
        return {}

    logging.info(f"🔢 Found {found_results} evaluation results")
    return aggregated


def calculate_averages(aggregated: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate average metrics for each question type and metric

    Returns:
        Dictionary with averaged results
    """
    averaged = {}

    for question_type, metrics in aggregated.items():
        averaged[question_type] = {}

        for metric_name, values in metrics.items():
            if values:
                avg_value = sum(values) / len(values)
                averaged[question_type][metric_name] = round(avg_value, 4)

                logging.info(
                    f"  {question_type} - {metric_name}: "
                    f"avg={avg_value:.4f} (n={len(values)})"
                )

    return averaged


def save_averaged_results(results: Dict[str, Dict[str, float]], output_file: str):
    """Save averaged results to JSON file"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info(f"💾 Averaged results saved to: {output_file}")


def print_summary(results: Dict[str, Dict[str, float]]):
    """Print summary of averaged results"""
    print("\n" + "="*70)
    print("📊 Novel Dataset - Averaged Evaluation Results")
    print("="*70)

    for question_type in sorted(results.keys()):
        metrics = results[question_type]
        print(f"\n📌 {question_type}:")
        for metric_name in sorted(metrics.keys()):
            value = metrics[metric_name]
            print(f"   • {metric_name}: {value:.4f}")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Average evaluation results across all Novel subdatasets"
    )
    parser.add_argument(
        "--results_dir",
        required=True,
        help="Directory containing Novel-* subdirectories with evaluation results"
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Output file path for averaged results JSON"
    )

    args = parser.parse_args()

    logging.info("🚀 Starting Novel results averaging...")
    logging.info(f"📂 Results directory: {args.results_dir}")
    print()

    # Aggregate results
    aggregated = aggregate_results(args.results_dir)

    if not aggregated:
        logging.error("❌ No results to aggregate!")
        return 1

    # Calculate averages
    logging.info("\n📈 Calculating averages...")
    averaged = calculate_averages(aggregated)

    # Save results
    save_averaged_results(averaged, args.output_file)

    # Print summary
    print_summary(averaged)

    logging.info("✅ Averaging completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())

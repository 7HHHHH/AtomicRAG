#!/usr/bin/env python3
"""
Reprocess novel dataset from Parquet to JSON with proper sub-dataset separation
"""

import json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

def split_text(
    text: str,
    tokenizer: AutoTokenizer,
    chunk_token_size: int = 256,
    chunk_overlap_token_size: int = 32
) -> list[str]:
    """Split text into chunks based on token length with overlap"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    start = 0
    while start < len(tokens):
        end = min(start + chunk_token_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(
            chunk_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        chunks.append(chunk_text)
        if end == len(tokens):
            break
        start += chunk_token_size - chunk_overlap_token_size
    return chunks

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Reprocess dataset from Parquet to sub-dataset JSON structure")
    parser.add_argument('--dataset', type=str, required=True, choices=['novel', 'medical'],
                        help='Dataset to process (novel or medical)')
    args = parser.parse_args()

    dataset_name = args.dataset
    REPO_ROOT = Path(__file__).resolve().parents[1]

    # Input paths
    corpus_path = REPO_ROOT / f"datasets/atomicrag/abstract_qa/Corpus/{dataset_name}.parquet"
    questions_path = REPO_ROOT / f"datasets/atomicrag/abstract_qa/Questions/{dataset_name}_questions.parquet"

    # Output base path
    output_base = REPO_ROOT / f"dataset/{dataset_name}"
    output_base.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print("📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")

    # Process corpus
    print("\n📖 Loading corpus from Parquet...")
    corpus_dataset = load_dataset("parquet", data_files=str(corpus_path), split="train")

    print(f"✅ Found {len(corpus_dataset)} sub-datasets")

    # Process each sub-dataset
    for item in tqdm(corpus_dataset, desc="Processing sub-datasets"):
        corpus_name = item['corpus_name']
        context = item['context']

        # Create directory for this sub-dataset
        sub_dir = output_base / corpus_name
        sub_dir.mkdir(parents=True, exist_ok=True)

        # Split into chunks
        chunks = split_text(context, tokenizer, chunk_token_size=256, chunk_overlap_token_size=32)

        # Save chunks
        chunks_path = sub_dir / "chunks.json"
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        print(f"  ✅ {corpus_name}: {len(chunks)} chunks → {chunks_path}")

    # Process questions
    print("\n❓ Loading questions from Parquet...")
    questions_dataset = load_dataset("parquet", data_files=str(questions_path), split="train")

    # Group questions by source
    questions_by_source = {}
    for item in questions_dataset:
        source = item['source']
        if source not in questions_by_source:
            questions_by_source[source] = []

        questions_by_source[source].append({
            'id': item['id'],
            'question_type': item['question_type'],
            'question': item['question'],
            'answer': item['answer'],
            'source': item['source'],
            'evidence': item['evidence']
        })

    # Save questions for each sub-dataset
    print(f"\n📝 Distributing questions to {len(questions_by_source)} sub-datasets...")
    for source, questions in questions_by_source.items():
        sub_dir = output_base / source

        if not sub_dir.exists():
            print(f"  ⚠️  Warning: No chunks found for {source}, skipping questions")
            continue

        questions_path = sub_dir / "questions.json"
        with open(questions_path, 'w', encoding='utf-8') as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)

        print(f"  ✅ {source}: {len(questions)} questions → {questions_path}")

    print("\n✨ Dataset reprocessing completed!")
    print(f"📁 Output directory: {output_base}")
    print(f"📊 Total sub-datasets: {len(corpus_dataset)}")

if __name__ == "__main__":
    main()

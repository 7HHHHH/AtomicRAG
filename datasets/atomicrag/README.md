# AtomicRAG Dataset Layout

This directory stores prepared QA data used by AtomicRAG examples and benchmark scripts.

- `abstract_qa/`: chunk-style corpora and question tables for broader reading-comprehension tasks.
- `precise_qa/`: factoid and multi-hop QA files used with the concise `precise` prompt template.

The main benchmark runner uses the full corpus tree under `dataset/<subset>/<corpus_name>/`, where each corpus directory contains `chunks.json` and `questions.json`. Both `dataset/` and `datasets/` are intended to be versioned in the public release.

Prompt templates live under `atomicrag/prompts/templates/`. Use `--qa_prompt_template precise` for short-answer benchmarks such as HotpotQA, MuSiQue, and 2WikiMultiHopQA.

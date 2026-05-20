# AtomicRAG Configuration

This directory contains runtime configuration for AtomicRAG.

## Files

- `config.json`: application-level defaults such as LLM and evaluation concurrency.
- `llm.env.example`: safe template for local API credentials.
- `llm.env`: your local credential file. This file is ignored by git and should never be committed.

## Setup

```bash
cp configs/atomicrag/llm.env.example configs/atomicrag/llm.env
```

Then edit `configs/atomicrag/llm.env` with your own OpenAI-compatible endpoint and key.

The runner scripts load this file automatically when it exists. You can also set the same variables directly in your shell:

```bash
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_API_KEY=your_api_key_here
export OPENAI_API_KEY="$LLM_API_KEY"
```

`config.json` currently sets both generation and evaluation concurrency to `200`, matching the released experiment configuration.

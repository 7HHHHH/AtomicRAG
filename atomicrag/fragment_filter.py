"""
Knowledge Fragment LLM Filter
Removes irrelevant fragments before QA generation
"""

import asyncio
import json
import logging
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple

from pydantic import BaseModel, Field, TypeAdapter
from tqdm import tqdm

logger = logging.getLogger(__name__)


class KeptFragmentIndices(BaseModel):
    """Data model for filtered fragment indices returned by the LLM."""
    keep_indices: List[int] = Field(
        description="Indices of fragments to retain (0-based)"
    )


@dataclass
class FragmentFilterRequest:
    """Request payload for concurrent fragment filtering."""
    request_id: int
    question: str
    fragments: List[str]


class FragmentFilter:
    """
    LLM-based filter for knowledge fragments.

    Filters out irrelevant fragments before QA generation to reduce noise.
    Uses similar architecture as DSPyFilter for facts.
    """

    def __init__(self, rag_system):
        """
        Initialize FragmentFilter

        Args:
            rag_system: AtomicRAG instance providing config and LLM
        """
        self.llm_model = rag_system.llm_model
        self.global_config = rag_system.global_config
        self.model_name = rag_system.global_config.llm_name
        self.default_gen_kwargs = {}

        # Metadata collector for token statistics
        self.metadata_list = []

        # Templates
        self.input_template = (
            "[[ ## question ## ]]\n{question}\n\n"
            "[[ ## fragments_before_filter ## ]]\n{fragments_before_filter}\n\n"
            "Respond with the field `[[ ## kept_fragment_ids ## ]]` "
            "(formatted as valid JSON), then `[[ ## completed ## ]]`."
        )
        self.output_template = (
            "[[ ## kept_fragment_ids ## ]]\n{kept_fragment_ids}\n\n"
            "[[ ## completed ## ]]"
        )

        self.message_template = self._make_template()

    def _make_template(self) -> List[Dict[str, str]]:
        """Build message template with system prompt and few-shot examples"""

        system_prompt = """Your input fields are:
1. `question` (str): The user's question
2. `fragments_before_filter` (str): Knowledge fragments to be filtered

Your output fields are:
1. `kept_fragment_ids` (KeptFragmentIndices): JSON object describing which fragment IDs to keep

All interactions will be structured as:

[[ ## question ## ]]
{question}

[[ ## fragments_before_filter ## ]]
{fragments_before_filter}

[[ ## kept_fragment_ids ## ]]
{kept_fragment_ids}  # JSON: {"keep_indices": [0, 2, ...]}

[[ ## completed ## ]]

Objective: Filter knowledge fragments to keep ONLY those directly relevant to answering the question.
Each fragment includes an `id` and the `text`. Choose the ids that should be kept.

KEEP fragments that:
- Directly answer the question
- Provide essential context for understanding the answer
- Contain key facts needed for reasoning

REMOVE fragments that:
- Discuss tangential topics
- Mention similar but different entities
- Provide generic background not needed for this specific question

Return kept fragment ids in JSON format: {"keep_indices": [id1, id2, ...]}
Return an empty array if none are relevant: {"keep_indices": []}
Do NOT invent new ids. Only choose from the provided fragment ids.
"""

        message_template = [{"role": "system", "content": system_prompt}]

        # Few-shot examples
        examples = [
            {
                "question": "What is the capital of France?",
                "fragments_before_filter": json.dumps({
                    "fragments": [
                        {"id": 0, "text": "France is a country in Western Europe."},
                        {"id": 1, "text": "Paris is the capital and most populous city of France."},
                        {"id": 2, "text": "The French Revolution began in 1789."},
                        {"id": 3, "text": "Paris has been the capital since 987 AD."}
                    ]
                }),
                "kept_fragment_ids": json.dumps({
                    "keep_indices": [1, 3]
                })
            },
            {
                "question": "When did World War II end?",
                "fragments_before_filter": json.dumps({
                    "fragments": [
                        {"id": 0, "text": "World War II lasted from 1939 to 1945."},
                        {"id": 1, "text": "Japan formally surrendered on September 2, 1945."},
                        {"id": 2, "text": "The war had devastating effects on global economy."},
                        {"id": 3, "text": "World War II officially ended with Japan's surrender in 1945."}
                    ]
                }),
                "kept_fragment_ids": json.dumps({
                    "keep_indices": [0, 1, 3]
                })
            }
        ]

        for example in examples:
            message_template.append({
                "role": "user",
                "content": self.input_template.format(
                    question=example["question"],
                    fragments_before_filter=example["fragments_before_filter"]
                )
            })
            message_template.append({
                "role": "assistant",
                "content": self.output_template.format(
                    kept_fragment_ids=example["kept_fragment_ids"]
                )
            })

        return message_template

    def _build_semaphore(self) -> Tuple[asyncio.Semaphore, int]:
        """直接使用全局 max_concurrency 配置"""
        limit = max(1, getattr(self.global_config, "max_concurrency", 1))
        return asyncio.Semaphore(limit), limit

    def _parse_response(self, response: str, num_fragments: int) -> List[int]:
        """
        Parse LLM response to extract kept fragment indices

        Args:
            response: LLM response string
            num_fragments: Number of original fragments (for validation & fallback)

        Returns:
            List of fragment indices to keep (returns full index list on parse failure)
        """
        sections = [(None, [])]
        field_pattern = re.compile(r'\[\[ ## (\w+) ## \]\]')

        for line in response.splitlines():
            match = field_pattern.match(line.strip())
            if match:
                sections.append((match.group(1), []))
            else:
                sections[-1][1].append(line)

        sections = [(k, "\n".join(v).strip()) for k, v in sections]

        for k, value in sections:
            if k == "kept_fragment_ids":
                try:
                    # Try JSON parsing
                    try:
                        parsed_value = json.loads(value)
                    except json.JSONDecodeError:
                        # Fallback to ast.literal_eval
                        import ast
                        parsed_value = ast.literal_eval(value)

                    # Validate with Pydantic
                    kept_model = TypeAdapter(KeptFragmentIndices).validate_python(
                        parsed_value
                    )
                    seen = set()
                    validated_indices = []
                    for idx in kept_model.keep_indices:
                        if not isinstance(idx, int):
                            logger.warning(f"Skipping non-integer index from fragment filter output: {idx}")
                            continue
                        if idx < 0 or idx >= num_fragments:
                            logger.warning(f"Skipping out-of-range index from fragment filter output: {idx}")
                            continue
                        if idx in seen:
                            continue
                        seen.add(idx)
                        validated_indices.append(idx)
                    return validated_indices

                except Exception as e:
                    logger.error(
                        f"Error parsing kept_fragment_ids: {e}\n"
                        f"Value:\n```\n{value}\n```"
                    )
                    logger.warning("Parse failed - returning all fragment indices as fallback")
                    return list(range(num_fragments))

        # If no kept_fragment_ids section found, return original indices
        logger.warning("No kept_fragment_ids section found - returning all fragment indices")
        return list(range(num_fragments))

    async def _llm_call_async(self, question: str, fragments: List[str]) -> str:
        """
        Call LLM to filter fragments

        Args:
            question: User's question
            fragments: List of fragment texts to filter

        Returns:
            LLM response string
        """
        # Build input with explicit fragment IDs to avoid ambiguity
        fragments_input = json.dumps({
            "fragments": [
                {"id": idx, "text": fragment}
                for idx, fragment in enumerate(fragments)
            ]
        })

        messages = deepcopy(self.message_template)
        messages.append({
            "role": "user",
            "content": self.input_template.format(
                question=question,
                fragments_before_filter=fragments_input
            )
        })

        # Set max tokens in a per-call copy to stay thread-safe
        gen_kwargs = dict(self.default_gen_kwargs)
        gen_kwargs['max_completion_tokens'] = 1024

        # Call LLM
        response, metadata, _ = await self.llm_model.ainfer(
            messages=messages,
            model=self.model_name,
            **gen_kwargs
        )
        # Collect metadata for statistics
        if metadata:
            self.metadata_list.append(metadata)

        return response

    async def filter_fragments_async(
        self,
        query: str,
        fragments: List[str]
    ) -> tuple[List[str], List[int]]:
        """
        Filter fragments using LLM

        Args:
            query: User's question
            fragments: List of retrieved fragment texts

        Returns:
            Tuple of (filtered_fragments, original_indices)
            - filtered_fragments: List of filtered fragments (LLM decides how many to keep)
            - original_indices: List of indices mapping back to original fragments array
        """
        if not fragments:
            return [], []

        try:
            # Call LLM
            response = await self._llm_call_async(query, fragments)

            # Parse response (with fallback to original indices on failure)
            kept_indices = self._parse_response(response, len(fragments))

            if not kept_indices:
                logger.debug("Fragment filter kept no fragments; downstream QA will receive empty context.")
                return [], []

            filtered_fragments = [fragments[idx] for idx in kept_indices]

            logger.debug(
                f"Fragment filter: {len(fragments)} → {len(filtered_fragments)} "
                f"(removed {len(fragments) - len(filtered_fragments)}) "
                f"indices: {kept_indices}"
            )

            return filtered_fragments, kept_indices

        except Exception as e:
            logger.error(f"Fragment filtering failed: {e}")
            logger.warning("Returning original fragments due to filtering error")
            # Return all fragments with their original indices
            return fragments, list(range(len(fragments)))

    async def filter_fragments_batch_async(
        self,
        requests: List[FragmentFilterRequest]
    ) -> Dict[int, tuple[List[str], List[int]]]:
        """
        Run fragment filtering concurrently for multiple queries using asyncio.
        使用全局 max_concurrency 配置控制并发。
        """
        if not requests:
            return {}

        semaphore, worker_limit = self._build_semaphore()
        logger.info(f"FragmentFilter: running {len(requests)} tasks with {worker_limit} workers (async)")
        fragments_lookup = {req.request_id: req.fragments for req in requests}
        results: Dict[int, tuple[List[str], List[int]]] = {}

        task_timeout = 60.0  # 60s timeout per task

        async def worker(req: FragmentFilterRequest):
            async with semaphore:
                try:
                    results[req.request_id] = await asyncio.wait_for(
                        self.filter_fragments_async(
                            query=req.question,
                            fragments=req.fragments
                        ),
                        timeout=task_timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Fragment filter task {req.request_id} timed out")
                    fallback_fragments = fragments_lookup[req.request_id]
                    results[req.request_id] = (
                        fallback_fragments,
                        list(range(len(fallback_fragments)))
                    )
                except Exception as exc:
                    logger.error(f"Fragment filter task {req.request_id} failed: {exc}")
                    fallback_fragments = fragments_lookup[req.request_id]
                    results[req.request_id] = (
                        fallback_fragments,
                        list(range(len(fallback_fragments)))
                    )

        tasks = [asyncio.create_task(worker(req)) for req in requests]
        progress = tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Fragment Filter",
            mininterval=0.5,
            leave=True
        )
        for completed in progress:
            await completed

        return results

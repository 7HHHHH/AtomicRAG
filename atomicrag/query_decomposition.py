"""
Query Decomposition Module for AtomicRAG

This module provides question decomposition functionality to improve retrieval quality
by breaking down complex questions into simpler sub-questions.

"""

import json
import logging
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from tqdm import tqdm

from .llm import BaseLLM
from .utils.llm_utils import TextChatMessage
from .prompts.prompt_template_manager import PromptTemplateManager

logger = logging.getLogger(__name__)


# ==================== Data Structures ====================

@dataclass
class SubQuestion:
    """Represents a single sub-question in the decomposition"""
    id: int
    question: str
    focus: str  # "entity" | "relationship" | "reasoning" | "context"
    docs: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "id": self.id,
            "question": self.question,
            "focus": self.focus,
            "docs": self.docs
        }


@dataclass
class DecompositionResult:
    """Complete decomposition result with metadata"""
    original_question: str
    question_type: Optional[str] = None
    is_decomposed: bool = False
    complexity_score: float = 0.0
    reasoning: str = ""
    sub_questions: List[SubQuestion] = field(default_factory=list)

    # Layered context from sub-questions
    sub_contexts: List[List[str]] = field(default_factory=list)  # Each sub-question's contexts
    original_docs: List[str] = field(default_factory=list)       # Original question's documents (for decomposed queries)
    merged_context: List[str] = field(default_factory=list)      # Final merged context

    # Intermediate answers (for layered generation)
    sub_answers: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "original_question": self.original_question,
            "question_type": self.question_type,
            "is_decomposed": self.is_decomposed,
            "complexity_score": round(self.complexity_score, 4),
            "reasoning": self.reasoning,
            "sub_questions": [sq.to_dict() for sq in self.sub_questions],
            "num_sub_questions": len(self.sub_questions),
            "sub_contexts": self.sub_contexts,
            "original_docs": self.original_docs,
            "merged_context": self.merged_context,
            "sub_answers": self.sub_answers
        }


# ==================== Query Decomposer Class ====================

class QueryDecomposer:
    """
    Handles dynamic question decomposition with LLM-based analysis.

    Workflow:
    1. Single LLM call scores complexity + emits sub-questions when needed
    2. Support for parallel retrieval of sub-questions
    3. Layered context integration
    """

    def __init__(self,
                 llm_model: BaseLLM,
                 max_sub_questions: int = 5,
                 complexity_threshold: float = 5.0,
                 prompt_template_manager: Optional[PromptTemplateManager] = None):
        """
        Initialize the QueryDecomposer.

        Args:
            llm_model: The LLM instance from AtomicRAG
            max_sub_questions: Maximum number of sub-questions to generate (default: 5)
            complexity_threshold: Threshold for decomposition decision (default: 5.0)
            prompt_template_manager: Optional prompt manager (defaults to a new instance)
        """
        self.llm_model = llm_model
        self.max_sub_questions = max_sub_questions
        self.complexity_threshold = complexity_threshold
        self.prompt_template_manager = prompt_template_manager or PromptTemplateManager()

        # Metadata collector for token statistics
        self.metadata_list = []

        logger.info(f"Initialized QueryDecomposer with max_sub_questions={max_sub_questions}, "
                   f"complexity_threshold={complexity_threshold}")

    def _build_analysis_messages(self,
                                 question: str,
                                 question_type: Optional[str]) -> List[TextChatMessage]:
        prompt = self.prompt_template_manager.render(
            name="question_analysis",
            question=question,
            question_type=question_type or "Unknown",
            max_sub_questions=self.max_sub_questions
        )
        return [TextChatMessage(role="user", content=prompt)]

    def merge_contexts(self,
                      sub_contexts: List[List[str]],
                      strategy: str = "deduplicate") -> List[str]:
        """
        Merge contexts from multiple sub-questions.

        Args:
            sub_contexts: List of context lists from each sub-question
            strategy: Merging strategy - "deduplicate" (default) or "weighted"

        Returns:
            Merged context list
        """
        if strategy == "deduplicate":
            # Simple deduplication while preserving order
            seen = set()
            merged = []
            for contexts in sub_contexts:
                for ctx in contexts:
                    if ctx not in seen:
                        seen.add(ctx)
                        merged.append(ctx)

            logger.info(f"Merged {sum(len(c) for c in sub_contexts)} contexts into {len(merged)} unique contexts")
            return merged

        elif strategy == "weighted":
            # Placeholder for future weighted merging
            # Could consider: relevance scores, sub-question importance, etc.
            return self.merge_contexts(sub_contexts, strategy="deduplicate")

        else:
            raise ValueError(f"Unknown merging strategy: {strategy}")

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response, handling various formats.

        Args:
            response_text: Raw LLM response

        Returns:
            Parsed JSON dictionary
        """
        # Try direct parsing
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code blocks
        if "```json" in response_text:
            try:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            except (IndexError, json.JSONDecodeError):
                pass

        # Try extracting any JSON object
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # If all parsing fails, raise error
        raise ValueError(f"Could not parse JSON from response: {response_text[:200]}...")

    # ==================== Async Methods for Concurrent Processing ====================

    async def analyze_question_async(self,
                                     question: str,
                                     question_type: Optional[str] = None) -> Tuple[bool, float, str, List[SubQuestion]]:
        """Run a single LLM call that evaluates complexity and generates sub-questions.

        Returns:
            Tuple of (needs_decomposition, complexity_score, reasoning, sub_questions)
        """
        try:
            messages = self._build_analysis_messages(question, question_type)
            response_message, metadata, _ = await self.llm_model.ainfer(messages)
            if metadata:
                self.metadata_list.append(metadata)

            if isinstance(response_message, str):
                response_text = response_message.strip()
            else:
                response_text = response_message.content.strip()

            result = self._parse_json_response(response_text)

            needs_decomposition = bool(result.get("needs_decomposition", False))
            complexity_score = float(result.get("complexity_score", 0.0))
            reasoning = result.get("reasoning", "")

            sub_questions_data = result.get("sub_questions", []) or []
            sub_questions: List[SubQuestion] = []
            for sq_data in sub_questions_data[:self.max_sub_questions]:
                if not isinstance(sq_data, dict):
                    continue
                sub_q = SubQuestion(
                    id=int(sq_data.get("id", len(sub_questions) + 1)),
                    question=sq_data.get("question", "").strip(),
                    focus=sq_data.get("focus", "reasoning")
                )
                sub_questions.append(sub_q)

            if complexity_score < self.complexity_threshold and needs_decomposition:
                needs_decomposition = False
                sub_questions = []
                reasoning += f" (Below threshold {self.complexity_threshold})"

            if needs_decomposition and not sub_questions:
                reasoning += " (LLM returned no sub-questions)"
                needs_decomposition = False

            return needs_decomposition, complexity_score, reasoning, sub_questions

        except Exception as e:
            logger.error(f"Error in question analysis: {e}", exc_info=True)
            return False, 0.0, f"Error during analysis: {str(e)}", []

    async def process_query_async(self,
                                 question: str,
                                 question_type: Optional[str] = None) -> DecompositionResult:
        """
        Async version of process_query for concurrent processing.

        This method processes a single query asynchronously, making it suitable
        for batch processing with asyncio.gather().

        Args:
            question: The original question
            question_type: Optional question type

        Returns:
            DecompositionResult with all metadata
        """
        result = DecompositionResult(
            original_question=question,
            question_type=question_type
        )

        needs_decomposition, complexity_score, reasoning, sub_questions = await self.analyze_question_async(
            question, question_type
        )

        result.complexity_score = complexity_score
        result.reasoning = reasoning
        result.is_decomposed = needs_decomposition
        result.sub_questions = sub_questions

        if needs_decomposition and not sub_questions:
            result.is_decomposed = False
            logger.warning(f"Analysis indicated decomposition but no sub-questions were produced: {question}")

        return result

    async def process_queries_batch_async(self,
                                         questions: List[str],
                                         question_types: Optional[List[str]] = None,
                                         max_concurrent: Optional[int] = None,
                                         progress_callback: Optional[Callable[[int, DecompositionResult], None]] = None
                                         ) -> List[DecompositionResult]:
        """
        Process multiple queries concurrently using asyncio.

        Each query performs a single LLM call that returns both the
        complexity assessment and (optional) sub-questions, all under the
        same concurrency budget.

        Args:
            questions: List of questions to process
            question_types: Optional list of question types (same length as questions)
            max_concurrent: Optional limit on simultaneous query tasks (None = unlimited)
            progress_callback: Optional callable invoked as callback(index, result)

        Returns:
            List of DecompositionResult objects
        """
        if question_types is None:
            question_types = [None] * len(questions)

        if len(questions) != len(question_types):
            raise ValueError(f"Length mismatch: {len(questions)} questions vs {len(question_types)} types")

        if max_concurrent is not None and max_concurrent <= 0:
            raise ValueError(f"max_concurrent must be > 0, got {max_concurrent}")

        total_questions = len(questions)
        available_slot = getattr(self.llm_model.global_config, "max_concurrency", 1)
        if max_concurrent is not None:
            available_slot = min(available_slot, max_concurrent)

        concurrency_limit = available_slot if total_questions > 0 else 1
        concurrency_limit = max(1, min(concurrency_limit, total_questions if total_questions > 0 else 1))

        if total_questions == 0:
            return []

        results: List[DecompositionResult] = [
            DecompositionResult(original_question=q, question_type=q_type)
            for q, q_type in zip(questions, question_types)
        ]

        task_queue: asyncio.Queue[int] = asyncio.Queue()
        for idx in range(len(questions)):
            task_queue.put_nowait(idx)

        analysis_progress = tqdm(
            total=total_questions,
            desc="Query Analysis",
            leave=True,
            mininterval=0.5
        )

        async def worker(worker_id: int):
            while True:
                try:
                    index = await task_queue.get()
                except asyncio.CancelledError:
                    break

                try:
                    question = questions[index]
                    q_type = question_types[index]

                    try:
                        analysis_result = await self.analyze_question_async(question, q_type)
                    except Exception as worker_exc:  # pragma: no cover - defensive
                        logger.error(
                            f"Analysis worker {worker_id} failed for query index {index}: {worker_exc}",
                            exc_info=True
                        )
                        analysis_result = (False, 0.0, f"Worker error: {worker_exc}", [])

                    needs_decomposition, complexity_score, reasoning, sub_questions = analysis_result

                    result = results[index]
                    result.complexity_score = complexity_score
                    result.reasoning = reasoning
                    result.is_decomposed = needs_decomposition
                    result.sub_questions = sub_questions

                    if needs_decomposition and not sub_questions:
                        result.is_decomposed = False
                        result.reasoning += " (Analysis returned no sub-questions)"
                        logger.warning(
                            f"Analysis indicated decomposition but provided no sub-questions: {result.original_question}"
                        )

                    if progress_callback:
                        try:
                            progress_callback(index, result)
                        except Exception as callback_error:
                            logger.warning(f"Progress callback failed for item {index}: {callback_error}")

                    analysis_progress.update(1)
                finally:
                    task_queue.task_done()

        workers = [asyncio.create_task(worker(worker_idx)) for worker_idx in range(concurrency_limit)]

        await task_queue.join()

        analysis_progress.close()

        for worker_task in workers:
            worker_task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        return results

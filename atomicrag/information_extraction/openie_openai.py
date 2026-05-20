import asyncio
import concurrent.futures
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, TypedDict

from tqdm import tqdm

from ..prompts import PromptTemplateManager
from ..utils.logging_utils import get_logger
from ..utils.llm_utils import fix_broken_generated_json, filter_invalid_triples
from ..utils.misc_utils import TripleRawOutput, NerRawOutput, KnowledgeFragmentRawOutput, UnifiedRawOutput
from ..llm.openai_gpt import CacheOpenAI

logger = get_logger(__name__)


class ChunkInfo(TypedDict):
    num_tokens: int
    content: str
    chunk_order: List[Tuple]
    full_doc_ids: List[str]


@dataclass
class LLMInput:
    chunk_id: str
    input_message: List[Dict]


def _extract_ner_from_response(real_response):
    pattern = r'\{[^{}]*"named_entities"\s*:\s*\[[^\]]*\][^{}]*\}'
    match = re.search(pattern, real_response, re.DOTALL)
    if match is None:
        # If pattern doesn't match, return an empty list
        return []
    try:
        entities = eval(match.group())["named_entities"]
        # Filter out non-string entities (LLM may return dicts or other types)
        return [e for e in entities if isinstance(e, str)]
    except Exception:
        return []


class OpenIE:
    def __init__(self, llm_model: CacheOpenAI, global_config=None):
        # Init prompt template manager
        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        self.llm_model = llm_model
        # Set default config if not provided
        if global_config is None:
            from dataclasses import dataclass
            @dataclass
            class DefaultConfig:
                max_concurrency: int = 8
            self.global_config = DefaultConfig()
        else:
            self.global_config = global_config

        # Metadata collector for token statistics
        self.metadata_list = []

    def _disable_progress(self) -> bool:
        return self.global_config.max_concurrency > 2500

    def _get_semaphore(self) -> asyncio.Semaphore:
        max_concurrency = max(1, self.global_config.max_concurrency)
        return asyncio.Semaphore(max_concurrency)

    async def _with_semaphore(self, semaphore: asyncio.Semaphore, coro_func, *args, **kwargs):
        async with semaphore:
            return await coro_func(*args, **kwargs)

    async def _gather_with_progress(self, tasks: List[asyncio.Task], desc: str):
        if not tasks:
            return []

        results = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        num_cache_hit = 0

        pbar = tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=desc,
            disable=self._disable_progress(),
            mininterval=1.0,
            maxinterval=10.0
        )

        for future in pbar:
            result = await future
            results.append(result)
            metadata = getattr(result, "metadata", {}) or {}
            # Collect metadata for statistics
            if metadata:
                self.metadata_list.append(metadata)
            total_prompt_tokens += metadata.get('prompt_tokens', 0)
            total_completion_tokens += metadata.get('completion_tokens', 0)
            if metadata.get('cache_hit'):
                num_cache_hit += 1
            pbar.set_postfix({
                'total_prompt_tokens': total_prompt_tokens,
                'total_completion_tokens': total_completion_tokens,
                'num_cache_hit': num_cache_hit
            })

        logger.info(
            f"{desc} finished: {len(results)} chunks | prompt_tokens={total_prompt_tokens} "
            f"| completion_tokens={total_completion_tokens} | cache_hits={num_cache_hit}"
        )

        return results

    def _run_async(self, coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        def _run():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
                asyncio.set_event_loop(None)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(_run).result()

    async def ner_async(self, chunk_key: str, passage: str, use_cache: bool = True) -> NerRawOutput:
        # PREPROCESSING
        ner_input_message = self.prompt_template_manager.render(name='ner', passage=passage)
        raw_response = ""
        metadata = {}
        try:
            # LLM INFERENCE
            raw_response, metadata, cache_hit = await self.llm_model.ainfer(
                messages=ner_input_message,
                use_cache=use_cache,
            )
            metadata['cache_hit'] = cache_hit
            if metadata['finish_reason'] == 'length':
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response
            extracted_entities = _extract_ner_from_response(real_response)
            unique_entities = list(dict.fromkeys(extracted_entities))

        except Exception as e:
            # For any other unexpected exceptions, log them and return with the error message
            logger.warning(e)
            metadata.update({'error': str(e)})
            return NerRawOutput(
                chunk_id=chunk_key,
                response=raw_response,  # Store the error message in metadata
                unique_entities=[],
                metadata=metadata  # Store the error message in metadata
            )

        return NerRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            unique_entities=unique_entities,
            metadata=metadata
        )

    def ner(self, chunk_key: str, passage: str, use_cache: bool = True) -> NerRawOutput:
        return self._run_async(self.ner_async(chunk_key=chunk_key, passage=passage, use_cache=use_cache))

    async def triple_extraction_async(self, chunk_key: str, passage: str, named_entities: List[str]) -> TripleRawOutput:
        def _extract_triples_from_response(real_response):
            pattern = r'\{[^{}]*"triples"\s*:\s*\[[^\]]*\][^{}]*\}'
            match = re.search(pattern, real_response, re.DOTALL)
            if match is None:
                # If pattern doesn't match, return an empty list
                return []
            return eval(match.group())["triples"]

        # PREPROCESSING
        messages = self.prompt_template_manager.render(
            name='triple_extraction',
            passage=passage,
            named_entity_json=json.dumps({"named_entities": named_entities})
        )

        raw_response = ""
        metadata = {}
        try:
            # LLM INFERENCE
            raw_response, metadata, cache_hit = await self.llm_model.ainfer(
                messages=messages,
            )
            metadata['cache_hit'] = cache_hit
            if metadata.get('finish_reason') == 'length':
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response
            extracted_triples = _extract_triples_from_response(real_response)
            triplets = filter_invalid_triples(triples=extracted_triples)

        except Exception as e:
            logger.warning(f"Exception for chunk {chunk_key}: {e}")
            metadata.update({'error': str(e)})
            return TripleRawOutput(
                chunk_id=chunk_key,
                response=raw_response,
                metadata=metadata,
                triples=[]
            )

        # Success
        return TripleRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            metadata=metadata,
            triples=triplets
        )

    def triple_extraction(self, chunk_key: str, passage: str, named_entities: List[str]) -> TripleRawOutput:
        return self._run_async(self.triple_extraction_async(chunk_key=chunk_key, passage=passage, named_entities=named_entities))

    async def knowledge_fragment_extraction_async(self, chunk_key: str, passage: str, candidate_entities: List[str] = None) -> KnowledgeFragmentRawOutput:
        """
        Extract knowledge fragments from the given passage using LLM.

        Args:
            chunk_key: Unique identifier for the chunk
            passage: Text passage to extract knowledge fragments from
            candidate_entities: List of entities to choose from (typically from triple extraction)

        Returns:
            KnowledgeFragmentRawOutput: Contains extracted fragments and metadata
        """

        def _extract_fragments_from_response(real_response):
            # Extract knowledge fragments and fragment entities from JSON response
            # Expected format: {"knowledge_fragments": [...], "fragment_entities": [[...], [...]]}
            pattern = r'\{[^{}]*"knowledge_fragments"\s*:\s*\[[^\]]*\].*?"fragment_entities"\s*:\s*\[.*?\][^{}]*\}'
            match = re.search(pattern, real_response, re.DOTALL)
            if match is None:
                # Fallback to legacy "events" schema for compatibility
                legacy_pattern = r'\{[^{}]*"events"\s*:\s*\[[^\]]*\].*?"event_entities"\s*:\s*\[.*?\][^{}]*\}'
                match = re.search(legacy_pattern, real_response, re.DOTALL)
                if match is None:
                    legacy_events_only = r'\{[^{}]*"events"\s*:\s*\[[^\]]*\][^{}]*\}'
                    match = re.search(legacy_events_only, real_response, re.DOTALL)
                    if match is None:
                        return [], []
                    try:
                        parsed = eval(match.group())
                        return parsed.get("events", []), []
                    except Exception:
                        return [], []
            try:
                parsed = eval(match.group())
                if "knowledge_fragments" in parsed:
                    return parsed.get("knowledge_fragments", []), parsed.get("fragment_entities", [])
                return parsed.get("events", []), parsed.get("event_entities", [])
            except Exception:
                return [], []

        if candidate_entities is None:
            candidate_entities = []

        messages = self.prompt_template_manager.render(
            name='knowledge_fragment_extraction',
            passage=passage,
            candidate_entities=candidate_entities
        )

        raw_response = ""
        metadata = {}
        try:
            raw_response, metadata, cache_hit = await self.llm_model.ainfer(
                messages=messages,
            )
            metadata['cache_hit'] = cache_hit
            if metadata.get('finish_reason') == 'length':
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response
            extracted_fragments, extracted_fragment_entities = _extract_fragments_from_response(real_response)
            fragments = [fragment.strip() for fragment in extracted_fragments if fragment.strip()]
            fragment_entities = extracted_fragment_entities[:len(fragments)] if extracted_fragment_entities else [[] for _ in fragments]

        except Exception as e:
            logger.warning(f"Exception for chunk {chunk_key} in knowledge fragment extraction: {e}")
            metadata.update({'error': str(e)})
            return KnowledgeFragmentRawOutput(
                chunk_id=chunk_key,
                response=raw_response,
                metadata=metadata,
                knowledge_fragments=[],
                fragment_entities=[]
            )

        return KnowledgeFragmentRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            metadata=metadata,
            knowledge_fragments=fragments,
            fragment_entities=fragment_entities
        )

    def knowledge_fragment_extraction(self, chunk_key: str, passage: str, candidate_entities: List[str] = None) -> KnowledgeFragmentRawOutput:
        return self._run_async(self.knowledge_fragment_extraction_async(chunk_key=chunk_key, passage=passage, candidate_entities=candidate_entities))

    async def unified_triple_knowledge_fragment_extraction_async(self, chunk_key: str, passage: str, named_entities: List[str]) -> UnifiedRawOutput:
        """
        Extract both RDF triples and knowledge fragments from the given passage in a single LLM call.

        Args:
            chunk_key: Unique identifier for the chunk
            passage: Text passage to extract from
            named_entities: List of named entities to guide extraction

        Returns:
            UnifiedRawOutput: Contains extracted triples, fragments, and metadata
        """

        def _extract_unified_from_response(real_response):
            # Expected format: {"triples": [...], "knowledge_fragments": [...], "fragment_entities": [[...], [...]]}
            pattern = r'\{[^{}]*"triples"\s*:\s*\[[^\]]*\].*?"knowledge_fragments"\s*:\s*\[[^\]]*\].*?"fragment_entities"\s*:\s*\[.*?\][^{}]*\}'
            match = re.search(pattern, real_response, re.DOTALL)
            if match is None:
                try:
                    parsed = json.loads(real_response)
                    if "knowledge_fragments" in parsed:
                        return (
                            parsed.get("triples", []),
                            parsed.get("knowledge_fragments", []),
                            parsed.get("fragment_entities", [])
                        )
                    return (
                        parsed.get("triples", []),
                        parsed.get("events", []),
                        parsed.get("event_entities", [])
                    )
                except Exception:
                    legacy_pattern = r'\{[^{}]*"triples"\s*:\s*\[[^\]]*\].*?"events"\s*:\s*\[[^\]]*\].*?"event_entities"\s*:\s*\[.*?\][^{}]*\}'
                    match = re.search(legacy_pattern, real_response, re.DOTALL)
                    if match is None:
                        return [], [], []
            try:
                parsed = eval(match.group())
                if "knowledge_fragments" in parsed:
                    return (
                        parsed.get("triples", []),
                        parsed.get("knowledge_fragments", []),
                        parsed.get("fragment_entities", [])
                    )
                return (
                    parsed.get("triples", []),
                    parsed.get("events", []),
                    parsed.get("event_entities", [])
                )
            except Exception:
                return [], [], []

        messages = self.prompt_template_manager.render(
            name='unified_triple_knowledge_fragment_extraction',
            passage=passage,
            named_entity_json=json.dumps({"named_entities": named_entities})
        )

        raw_response = ""
        metadata = {}
        try:
            raw_response, metadata, cache_hit = await self.llm_model.ainfer(
                messages=messages,
            )
            metadata['cache_hit'] = cache_hit
            if metadata.get('finish_reason') == 'length':
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response

            extracted_triples, extracted_fragments, extracted_fragment_entities = _extract_unified_from_response(real_response)

            triples = filter_invalid_triples(triples=extracted_triples)
            fragments = [fragment.strip() for fragment in extracted_fragments if fragment.strip()]
            fragment_entities = extracted_fragment_entities[:len(fragments)] if extracted_fragment_entities else [[] for _ in fragments]

        except Exception as e:
            logger.warning(f"Exception for chunk {chunk_key} in unified extraction: {e}")
            metadata.update({'error': str(e)})
            return UnifiedRawOutput(
                chunk_id=chunk_key,
                response=raw_response,
                metadata=metadata,
                triples=[],
                knowledge_fragments=[],
                fragment_entities=[]
            )

        return UnifiedRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            metadata=metadata,
            triples=triples,
            knowledge_fragments=fragments,
            fragment_entities=fragment_entities
        )

    def unified_triple_knowledge_fragment_extraction(self, chunk_key: str, passage: str, named_entities: List[str]) -> UnifiedRawOutput:
        return self._run_async(
            self.unified_triple_knowledge_fragment_extraction_async(
                chunk_key=chunk_key,
                passage=passage,
                named_entities=named_entities
            )
        )

    async def openie_async(self, chunk_key: str, passage: str) -> Dict[str, Any]:
        ner_output = await self.ner_async(chunk_key=chunk_key, passage=passage)
        triple_output = await self.triple_extraction_async(chunk_key=chunk_key, passage=passage, named_entities=ner_output.unique_entities)

        # Extract entities from triples to use as candidates for knowledge fragment extraction
        triple_entities = set()
        for triple in triple_output.triples:
            if len(triple) >= 3:
                triple_entities.add(triple[0])  # subject
                triple_entities.add(triple[2])  # object

        fragment_output = await self.knowledge_fragment_extraction_async(chunk_key=chunk_key, passage=passage, candidate_entities=list(triple_entities))
        return {"ner": ner_output, "triplets": triple_output, "knowledge_fragments": fragment_output}

    def openie(self, chunk_key: str, passage: str) -> Dict[str, Any]:
        return self._run_async(self.openie_async(chunk_key=chunk_key, passage=passage))

    async def batch_openie_async(self, chunks: Dict[str, ChunkInfo]) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput], Dict[str, KnowledgeFragmentRawOutput]]:
        """
        Conduct batch OpenIE asynchronously using asyncio which includes NER, triple extraction, and knowledge fragment extraction.
        """
        chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}
        semaphore = self._get_semaphore()

        ner_tasks = [
            asyncio.create_task(self._with_semaphore(semaphore, self.ner_async, chunk_key, passage))
            for chunk_key, passage in chunk_passages.items()
        ]
        ner_results_list = await self._gather_with_progress(ner_tasks, desc="NER")

        triple_tasks = [
            asyncio.create_task(
                self._with_semaphore(
                    semaphore,
                    self.triple_extraction_async,
                    ner_result.chunk_id,
                    chunk_passages[ner_result.chunk_id],
                    ner_result.unique_entities
                )
            )
            for ner_result in ner_results_list
        ]
        triple_results_list = await self._gather_with_progress(triple_tasks, desc="Extracting triples")

        triple_results_dict = {res.chunk_id: res for res in triple_results_list}
        fragment_tasks = []
        for chunk_key, passage in chunk_passages.items():
            triple_result = triple_results_dict.get(chunk_key)
            candidate_entities = []
            if triple_result:
                triple_entities = set()
                for triple in triple_result.triples:
                    if len(triple) >= 3:
                        triple_entities.add(triple[0])
                        triple_entities.add(triple[2])
                candidate_entities = list(triple_entities)
            fragment_tasks.append(
                asyncio.create_task(
                    self._with_semaphore(
                        semaphore,
                        self.knowledge_fragment_extraction_async,
                        chunk_key,
                        passage,
                        candidate_entities
                    )
                )
            )
        fragment_results_list = await self._gather_with_progress(fragment_tasks, desc="Extracting knowledge fragments")

        ner_results_dict = {res.chunk_id: res for res in ner_results_list}
        fragment_results_dict = {res.chunk_id: res for res in fragment_results_list}

        return ner_results_dict, triple_results_dict, fragment_results_dict

    def batch_openie(self, chunks: Dict[str, ChunkInfo]) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput], Dict[str, KnowledgeFragmentRawOutput]]:
        return self._run_async(self.batch_openie_async(chunks))

    async def batch_unified_openie_async(self, chunks: Dict[str, ChunkInfo]) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput], Dict[str, KnowledgeFragmentRawOutput]]:
        """
        Conduct batch OpenIE using unified extraction (NER + unified triple/fragment extraction in one call).
        """
        chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}
        semaphore = self._get_semaphore()

        ner_tasks = [
            asyncio.create_task(self._with_semaphore(semaphore, self.ner_async, chunk_key, passage))
            for chunk_key, passage in chunk_passages.items()
        ]
        ner_results_list = await self._gather_with_progress(ner_tasks, desc="NER")

        unified_tasks = [
            asyncio.create_task(
                self._with_semaphore(
                    semaphore,
                    self.unified_triple_knowledge_fragment_extraction_async,
                    ner_result.chunk_id,
                    chunk_passages[ner_result.chunk_id],
                    ner_result.unique_entities
                )
            )
            for ner_result in ner_results_list
        ]
        unified_results_list = await self._gather_with_progress(unified_tasks, desc="Unified triple+fragment extraction")

        ner_results_dict = {res.chunk_id: res for res in ner_results_list}
        triple_results_dict = {}
        fragment_results_dict = {}

        for unified_result in unified_results_list:
            # Create TripleRawOutput from unified result
            triple_results_dict[unified_result.chunk_id] = TripleRawOutput(
                chunk_id=unified_result.chunk_id,
                response=unified_result.response,
                metadata=unified_result.metadata,
                triples=unified_result.triples
            )

            # Create KnowledgeFragmentRawOutput from unified result
            fragment_results_dict[unified_result.chunk_id] = KnowledgeFragmentRawOutput(
                chunk_id=unified_result.chunk_id,
                response=unified_result.response,
                metadata=unified_result.metadata,
                knowledge_fragments=unified_result.knowledge_fragments,
                fragment_entities=unified_result.fragment_entities
            )

        return ner_results_dict, triple_results_dict, fragment_results_dict

    def batch_unified_openie(self, chunks: Dict[str, ChunkInfo]) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput], Dict[str, KnowledgeFragmentRawOutput]]:
        return self._run_async(self.batch_unified_openie_async(chunks))

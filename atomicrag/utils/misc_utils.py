from dataclasses import dataclass
from hashlib import md5
from typing import Dict, Any, List, Tuple, Literal, Union, Optional
import numpy as np
import re
import logging

from .typing import Triple
from .llm_utils import filter_invalid_triples

logger = logging.getLogger(__name__)

@dataclass
class NerRawOutput:
    chunk_id: str
    response: str
    unique_entities: List[str]
    metadata: Dict[str, Any]


@dataclass
class TripleRawOutput:
    chunk_id: str
    response: str
    triples: List[List[str]]
    metadata: Dict[str, Any]

@dataclass
class KnowledgeFragmentRawOutput:
    chunk_id: str
    response: str
    knowledge_fragments: List[str]  # List of knowledge fragments extracted from the chunk
    fragment_entities: List[List[str]]  # List of entity lists, each corresponding to a fragment
    metadata: Dict[str, Any]

@dataclass
class UnifiedRawOutput:
    chunk_id: str
    response: str
    triples: List[List[str]]  # RDF triples extracted from the chunk
    knowledge_fragments: List[str]  # Knowledge fragments extracted from the chunk
    fragment_entities: List[List[str]]  # Entity lists corresponding to each fragment
    metadata: Dict[str, Any]

@dataclass
class LinkingOutput:
    score: np.ndarray
    type: Literal['node', 'dpr']

@dataclass
class QuerySolution:
    question: str
    docs: List[str]
    answer: str = None
    gold_answers: List[str] = None
    gold_docs: Optional[List[str]] = None
    decomposition_metadata: Optional[Dict[str, Any]] = None  # Added for question decomposition


    def to_dict(self):
        result = {
            "question": self.question,
            "answer": self.answer,
            "gold_answers": self.gold_answers,
            "docs": self.docs,  # Use all retrieved docs, not just first 5
            "gold_docs": self.gold_docs,
        }

        # Add decomposition metadata if available
        if self.decomposition_metadata is not None:
            result["decomposition_metadata"] = self.decomposition_metadata

        return result

def text_processing(text):
    if isinstance(text, list):
        return [text_processing(t) for t in text]
    if not isinstance(text, str):
        text = str(text)
    return re.sub('[^A-Za-z0-9 ]', ' ', text.lower()).strip()

def reformat_openie_results(corpus_openie_results) -> (Dict[str, NerRawOutput], Dict[str, TripleRawOutput], Dict[str, KnowledgeFragmentRawOutput]):

    ner_output_dict = {
        chunk_item['idx']: NerRawOutput(
            chunk_id=chunk_item['idx'],
            response=None,
            metadata={},
            unique_entities=list(np.unique(chunk_item['extracted_entities']))
        )
        for chunk_item in corpus_openie_results
    }
    triple_output_dict = {
        chunk_item['idx']: TripleRawOutput(
            chunk_id=chunk_item['idx'],
            response=None,
            metadata={},
            triples=filter_invalid_triples(triples=chunk_item['extracted_triples'])
        )
        for chunk_item in corpus_openie_results
    }
    fragment_output_dict = {
        chunk_item['idx']: KnowledgeFragmentRawOutput(
            chunk_id=chunk_item['idx'],
            response=None,
            metadata={},
            knowledge_fragments=chunk_item.get('extracted_knowledge_fragments', chunk_item.get('extracted_events', [])),
            fragment_entities=chunk_item.get('extracted_fragment_entities', chunk_item.get('extracted_event_entities', []))
        )
        for chunk_item in corpus_openie_results
    }

    return ner_output_dict, triple_output_dict, fragment_output_dict

def extract_entity_nodes(chunk_triples: List[List[Triple]]) -> (List[str], List[List[str]]):
    chunk_triple_entities = []  # a list of lists of unique entities from each chunk's triples
    for triples in chunk_triples:
        triple_entities = set()
        for t in triples:
            if len(t) == 3:
                triple_entities.update([t[0], t[2]])
            else:
                logger.warning(f"During graph construction, invalid triple is found: {t}")
        chunk_triple_entities.append(list(triple_entities))
    graph_nodes = list(np.unique([ent for ents in chunk_triple_entities for ent in ents]))
    return graph_nodes, chunk_triple_entities

def flatten_facts(chunk_triples: List[Triple]) -> List[Triple]:
    graph_triples = []  # a list of unique relation triple (in tuple) from all chunks
    for triples in chunk_triples:
        graph_triples.extend([tuple(t) for t in triples])
    graph_triples = list(set(graph_triples))
    return graph_triples

def min_max_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val

    # Handle the case where all values are the same (range is zero)
    if range_val == 0:
        return np.ones_like(x)  # Return an array of ones with the same shape as x

    return (x - min_val) / range_val

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute the MD5 hash of the given content string and optionally prepend a prefix.

    Args:
        content (str): The input string to be hashed.
        prefix (str, optional): A string to prepend to the resulting hash. Defaults to an empty string.

    Returns:
        str: A string consisting of the prefix followed by the hexadecimal representation of the MD5 hash.
    """
    return prefix + md5(content.encode()).hexdigest()

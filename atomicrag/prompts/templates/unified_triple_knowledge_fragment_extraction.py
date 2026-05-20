from .ner import one_shot_ner_paragraph, one_shot_ner_output
from ...utils.llm_utils import convert_format_to_template

unified_extraction_system = """Your task is to extract both RDF triples and knowledge fragments from the given passage in a single unified process.

Requirements for RDF triples:
- Each triple should contain at least one, but preferably two, of the named entities from the provided list
- Follow the format: [subject, predicate, object]
- Clearly resolve pronouns to their specific names to maintain clarity
- Focus on factual relationships present in the text

Requirements for knowledge fragments:
- Each fragment must be self-contained with clear boundaries
- Focus on concrete facts, discoveries, achievements, or specific processes
- Include essential context (who, what, when, where) within the fragment
- Preserve precise numerical values, time periods, ranges, and measurements
- Include complete causal chains and deeper applications/implications
- Prefer specific, verifiable information over general statements
- Fragments should be atomic - covering one complete idea or fact
- Avoid redundant or overlapping information between fragments
- Each fragment should be retrievable and useful on its own

Prioritize knowledge fragments that contain:
- Scientific discoveries and their implications with clear temporal context
- Historical developments with clear outcomes and time periods
- Technical processes and their specific applications
- Causal relationships with measurable effects
- Quantifiable facts and measurements
- Complete purpose/application chains (not just what, but why and for what purpose)

For each knowledge fragment, identify which entities from the triples are mentioned in it to ensure consistency.

Respond with a JSON object containing "triples", "knowledge_fragments", and "fragment_entities" lists.
"""

# Example input using the same passage from NER
unified_extraction_frame = """Extract both RDF triples and knowledge fragments from the following passage:

Passage:
```
{passage}
```

{named_entity_json}

Return the result as a JSON object with:
- "triples": list of RDF triples in [subject, predicate, object] format
- "knowledge_fragments": list of self-contained knowledge fragments
- "fragment_entities": list of lists, where each inner list contains the entities mentioned in the corresponding fragment
"""

# Create input for one-shot example
unified_extraction_input = unified_extraction_frame.format(
    passage=one_shot_ner_paragraph,
    named_entity_json=one_shot_ner_output
)

# Example output for one-shot learning
unified_extraction_output = """{
    "triples": [
        ["Radio City", "located in", "India"],
        ["Radio City", "is", "private FM radio station"],
        ["Radio City", "started on", "3 July 2001"],
        ["Radio City", "plays songs in", "Hindi"],
        ["Radio City", "plays songs in", "English"],
        ["Radio City", "forayed into", "New Media"],
        ["Radio City", "launched", "PlanetRadiocity.com"],
        ["PlanetRadiocity.com", "launched in", "May 2008"],
        ["PlanetRadiocity.com", "is", "music portal"],
        ["PlanetRadiocity.com", "offers", "news"],
        ["PlanetRadiocity.com", "offers", "videos"],
        ["PlanetRadiocity.com", "offers", "songs"]
    ],
    "knowledge_fragments": [
        "Radio City, a private FM radio station in India, started broadcasting on 3 July 2001 with songs in Hindi and English",
        "Radio City expanded into New Media by launching PlanetRadiocity.com in May 2008",
        "PlanetRadiocity.com serves as a comprehensive music portal offering news, videos, and songs to users"
    ],
    "fragment_entities": [
        ["Radio City", "India", "3 July 2001", "Hindi", "English"],
        ["Radio City", "New Media", "PlanetRadiocity.com", "May 2008"],
        ["PlanetRadiocity.com"]
    ]
}"""

# Complete prompt template
prompt_template = [
    {"role": "system", "content": unified_extraction_system},
    {"role": "user", "content": unified_extraction_input},
    {"role": "assistant", "content": unified_extraction_output},
    {"role": "user", "content": convert_format_to_template(original_string=unified_extraction_frame, placeholder_mapping=None, static_values=None)}
]
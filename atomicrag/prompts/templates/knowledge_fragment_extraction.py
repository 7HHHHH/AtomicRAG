

from ...utils.llm_utils import convert_format_to_template

# ============================================================================
# 改进的系统提示词（采用HyperGraphRAG的结构化方法）
# ============================================================================

knowledge_fragment_extraction_system = """---Goal---
Your task is to extract well-defined knowledge fragments from the given passage.
Each knowledge fragment should be a complete, self-contained piece of information that can be independently
understood and retrieved.

---Requirements---
Structure:
- Each knowledge fragment must be self-contained with clear boundaries
- Include essential context (who, what, when, where, why) within the fragment
- Focus on complete causal chains and relationships
- One complete idea or action per fragment

Content Quality:
- Prioritize specific, verifiable information over general statements
- Preserve precise numerical values, time periods, ranges, and measurements
- Include deeper applications, implications, or consequences
- Connect fragments through clear causal relationships

Knowledge Fragment Types to Extract:
- Scientific discoveries with their implications and temporal context
- Historical knowledge fragments with clear outcomes and time periods
- Technical processes and their specific applications
- Causal relationships with measurable effects
- Quantifiable facts, measurements, and milestones
- Complete purpose/application chains (what, why, and what for)

---Output Format---
Return a JSON object with:
- "knowledge_fragments": list of knowledge fragments (natural text, not structured fields)
- "fragment_entities": list of lists matching fragments, containing mentioned candidate entities
"""

# ============================================================================
# 改进的单发学习示例1（医学/科学领域）
# ============================================================================

one_shot_fragment_paragraph_1 = """The CRISPR-Cas9 gene editing technology was developed by Jennifer Doudna and Emmanuelle Charpentier in 2012.
This breakthrough allowed scientists to precisely edit DNA sequences in living cells.
In 2020, they received the Nobel Prize in Chemistry for this discovery.
CRISPR has since been used to treat genetic diseases including sickle cell disease and beta-thalassemia.
The first clinical trials for CRISPR-based treatments began in 2019 and showed promising results."""

one_shot_entities_1 = ["CRISPR-Cas9", "Jennifer Doudna", "Emmanuelle Charpentier", "2012", "DNA",
                       "Nobel Prize in Chemistry", "2020", "genetic diseases", "sickle cell disease",
                       "beta-thalassemia", "clinical trials", "2019"]

one_shot_knowledge_fragments_1 = """{
    "knowledge_fragments": [
        "CRISPR-Cas9 gene editing technology was developed by Jennifer Doudna and Emmanuelle Charpentier in 2012, enabling precise editing of DNA sequences in living cells",
        "Jennifer Doudna and Emmanuelle Charpentier received the Nobel Prize in Chemistry in 2020 for their development of CRISPR-Cas9 gene editing technology",
        "CRISPR-based treatments have been applied to genetic diseases including sickle cell disease and beta-thalassemia, with initial clinical trials beginning in 2019 showing promising results"
    ],
    "fragment_entities": [
        ["CRISPR-Cas9", "Jennifer Doudna", "Emmanuelle Charpentier", "2012", "DNA"],
        ["Jennifer Doudna", "Emmanuelle Charpentier", "Nobel Prize in Chemistry", "2020", "CRISPR-Cas9"],
        ["CRISPR", "sickle cell disease", "beta-thalassemia", "clinical trials", "2019"]
    ]
}"""

# ============================================================================
# 改进的单发学习示例2（历史知识片段领域）
# ============================================================================

one_shot_fragment_paragraph_2 = """The Panama Canal opened on August 15, 1914, after 10 years of construction.
It was built by the United States under the leadership of President Theodore Roosevelt.
The canal connects the Atlantic and Pacific Oceans, reducing shipping distances by over 8,000 miles.
Construction involved overcoming significant engineering challenges including tropical disease control.
The canal significantly boosted global trade and military strategic positioning."""

one_shot_entities_2 = ["Panama Canal", "August 15, 1914", "10 years", "United States", "Theodore Roosevelt",
                       "Atlantic Ocean", "Pacific Ocean", "8,000 miles", "tropical disease", "global trade"]

one_shot_knowledge_fragments_2 = """{
    "knowledge_fragments": [
        "The Panama Canal was completed and opened on August 15, 1914, after 10 years of construction by the United States under President Theodore Roosevelt",
        "The Panama Canal connects the Atlantic and Pacific Oceans, reducing shipping distances by over 8,000 miles and boosting global maritime trade",
        "Construction of the Panama Canal involved overcoming significant engineering challenges, particularly tropical disease control measures that affected workforce health and mortality"
    ],
    "fragment_entities": [
        ["Panama Canal", "August 15, 1914", "10 years", "United States", "Theodore Roosevelt"],
        ["Panama Canal", "Atlantic Ocean", "Pacific Ocean", "8,000 miles", "global trade"],
        ["Panama Canal", "construction", "tropical disease", "engineering challenges"]
    ]
}"""

# ============================================================================
# 改进的单发学习示例3（技术/商业领域）
# ============================================================================

one_shot_fragment_paragraph_3 = """Apple released the first iPhone on June 29, 2007, revolutionizing mobile technology.
The device was designed by Jony Ive and introduced features like touchscreen interface and App Store integration.
By 2010, the iPhone had captured 25% of the smartphone market.
The iPhone led to the development of iOS, which became the foundation for iPad and other Apple devices.
App developers could now create applications through the App Store, launched in July 2008, generating billions in revenue."""

one_shot_entities_3 = ["Apple", "iPhone", "June 29, 2007", "Jony Ive", "touchscreen interface", "App Store",
                       "2010", "25%", "smartphone market", "iOS", "iPad", "July 2008"]

one_shot_knowledge_fragments_3 = """{
    "knowledge_fragments": [
        "Apple released the first iPhone on June 29, 2007, featuring a touchscreen interface and revolutionary mobile design created by Jony Ive",
        "The App Store was launched by Apple in July 2008, enabling third-party developers to create and distribute applications for the iPhone platform",
        "By 2010, the iPhone had captured 25% of the smartphone market and led to the development of iOS, which became the foundation for iPad and other Apple products"
    ],
    "fragment_entities": [
        ["Apple", "iPhone", "June 29, 2007", "Jony Ive", "touchscreen interface"],
        ["App Store", "Apple", "July 2008", "developers", "applications"],
        ["iPhone", "2010", "25%", "smartphone market", "iOS", "iPad"]
    ]
}"""

# ============================================================================
# 改进的提示词框架（采用HyperGraphRAG的步骤化方法）
# ============================================================================

knowledge_fragment_extraction_frame = """---Goal---
Extract well-defined knowledge fragments from the given passage.

---Steps---
1. Read the passage carefully and identify distinct knowledge fragments or discoveries
   - Group related information into complete, self-contained fragments
   - Ensure each fragment can stand alone with sufficient context
   - Each fragment should represent one complete idea or action

2. For each identified knowledge fragment:
   - Write it as a complete, natural sentence (not bullet points or labels)
   - Include essential context: who, what, when, where, why/how
   - Verify it contains one complete idea
   - Check the completeness: does it make sense without external context?

3. Map fragments to candidate entities:
   - For each knowledge fragment, identify all mentioned candidate entities
   - Include direct mentions and implied references
   - Note the role each entity plays in the fragment

4. Return results in JSON format with proper structure

---Passage---
```
{passage}
```

---Candidate Entities (from previous extraction)---
{candidate_entities}

---Instructions for Output---
- "knowledge_fragments": List of complete knowledge fragments written as natural sentences
- "fragment_entities": Each inner list contains entities mentioned in the corresponding knowledge fragment

---Output Format (JSON)---
{{
    "knowledge_fragments": [...],
    "fragment_entities": [...]
}}
"""

# ============================================================================
# 多发学习示例准备（三个例子都包含）
# ============================================================================

def prepare_few_shot_examples():
    """
    准备多发学习的三个示例
    Returns: 格式化后的三个示例字符串
    """
    examples_text = f"""---Example 1: Scientific Discovery (CRISPR Gene Editing)---
{knowledge_fragment_extraction_frame.format(
    passage=one_shot_fragment_paragraph_1,
    candidate_entities=", ".join(one_shot_entities_1)
)}

Assistant Response:
{one_shot_knowledge_fragments_1}

---Example 2: Historical Knowledge Fragment (Panama Canal)---
{knowledge_fragment_extraction_frame.format(
    passage=one_shot_fragment_paragraph_2,
    candidate_entities=", ".join(one_shot_entities_2)
)}

Assistant Response:
{one_shot_knowledge_fragments_2}

---Example 3: Technology & Business Knowledge Fragment (iPhone Launch)---
{knowledge_fragment_extraction_frame.format(
    passage=one_shot_fragment_paragraph_3,
    candidate_entities=", ".join(one_shot_entities_3)
)}

Assistant Response:
{one_shot_knowledge_fragments_3}
"""
    return examples_text

# ============================================================================
# 改进的完整提示词模板（包含三个多发学习示例）
# ============================================================================

def get_prompt_template():
    """
    获取完整的提示词模板（包含三个多发学习示例）
    Returns:
        prompt_template: 完整的多轮对话模板
    """
    prompt_template = [
        {
            "role": "system",
            "content": knowledge_fragment_extraction_system
        },
        {
            "role": "user",
            "content": prepare_few_shot_examples()
        },
        {
            "role": "user",
            "content": convert_format_to_template(
                original_string=knowledge_fragment_extraction_frame,
                placeholder_mapping=None,
                static_values=None
            )
        }
    ]
    return prompt_template

# 默认使用包含三个示例的模板
prompt_template = get_prompt_template()

# ============================================================================
# 使用说明和最佳实践
# ============================================================================

"""
改进说明：

1. 结构化提示词设计（采自HyperGraphRAG）
   ✓ ---Goal--- 清晰的目标
   ✓ ---Steps--- 分步指导（4步）
   ✓ ---Requirements--- 具体要求
   ✓ 清晰的输出格式规范

2. 多发学习（Few-Shot Learning）- 三个示例全部包含
   ✓ 示例1：科学/医学领域（CRISPR基因编辑）- 5行
   ✓ 示例2：历史知识片段领域（巴拿马运河）- 5行
   ✓ 示例3：技术/商业领域（iPhone）- 5行
   ✓ 每个示例都包含3个高质量的知识片段
   ✓ 覆盖不同领域，提高LLM的泛化能力

3. 术语统一
   ✓ 内部统称为"知识片段"(knowledge fragments)
   ✓ JSON输出中"knowledge_fragments"字段统一表示片段
   ✓ 移除了所有打分字段

4. 完全保持兼容性
   ✓ JSON 输出格式完全兼容
   ✓ "knowledge_fragments" 字段结构清晰
   ✓ "fragment_entities" 结构保持一一对应

5. 使用方式

   # 使用默认模板（包含三个多发学习示例）
   from knowledge_fragment_extraction import prompt_template
   result = llm.chat(prompt_template, passage=user_passage, candidate_entities=user_entities)

   # 或者调用函数获取
   from knowledge_fragment_extraction import get_prompt_template
   prompt_template = get_prompt_template()
   result = llm.chat(prompt_template, passage=user_passage, candidate_entities=user_entities)

最佳实践建议：
- 三个多发学习示例覆盖科学、历史、技术三个不同领域
- 提示词会自动包含所有三个示例，无需手动选择
- 候选实体列表应来自之前的三元组抽取
- 保证每个知识片段都能独立理解
- 每个片段应包含完整的上下文（who, what, when, where, why）
- 在大规模应用前进行小规模验证

文件结构：
- knowledge_fragment_extraction_system: 系统提示词
- one_shot_fragment_paragraph_X: 示例X的输入文本
- one_shot_entities_X: 示例X的候选实体
- one_shot_knowledge_fragments_X: 示例X的输出
- prepare_few_shot_examples(): 生成三个多发学习示例
- get_prompt_template(): 获取完整提示词模板
- prompt_template: 默认提示词模板
"""

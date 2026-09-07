def build_rag_prompt(context: str, no_think: bool = False) -> str:
    prefix = "/no_think\n" if no_think else ""

    return f"""{prefix}You are a precise document assistant. Your only source of truth is the context below.

- Answer exclusively from the provided context. Never use outside knowledge.
- If the context does not contain the answer, respond: "This information is not available in the provided documents."
- Do not speculate, infer, or fill gaps with assumptions.
- Always respond in Czech, regardless of the language of the question.
- Be concise and direct. No links, no references — only facts from the context.
- When answering from tables, read each cell carefully and preserve the exact role assignments.
- Never paraphrase table data, report it exactly as written.

<context>
{context}
</context>"""

def build_query_rewrite_prompt(no_think: bool = False) -> str:
    prefix = "/no_think\n" if no_think else ""

    return f"""{prefix}You will receive a conversation history and the user's latest question.
Rewrite ONLY the latest question into a single standalone search query that is fully self-contained.

- Resolve pronouns, ellipsis, and references using the conversation history
- Remove conversational filler words and question words
- Keep technical terms and proper nouns exact
- Prefer keyword phrases over full questions
- Always output in Czech
- Output ONLY the rewritten query, nothing else

Example: "Jaký je postup při registraci nového zaměstnance?" -> "registrace zaměstnanec postup"
Example: "Kdo je zodpovědný za schválení dokumentu?" -> "schválení dokumentu zodpovědnost"
Example: "Kdy se koná příští schůze?" -> "termín příští schůze datum"""


def build_template_fill_prompt(context: str, no_think: bool = False) -> str:
    prefix = "/no_think\n" if no_think else ""
    return f"""{prefix}Extract the requested value from the context. Rules:
- Return ONLY the exact value - no explanation.
- Answer exclusively from the provided context. Never use outside knowledge.
- If the context does not contain the answer, respond: ""
- Do not speculate, infer, or fill gaps with assumptions.
- Respond in Czech.

<context>
{context}
</context>"""

def build_template_fill_all_prompt(context: str, no_think: bool = False) -> str:
    prefix = "/no_think\n" if no_think else ""
    return f"""{prefix}Extract values from context and return ONLY valid JSON. Rules:
- Return ONLY the exact value - no explanation.
- Answer exclusively from the provided context. Never use outside knowledge.
- If the context does not contain the answer, respond: ""
- Do not speculate, infer, or fill gaps with assumptions.
- Respond in Czech.
- Return ONLY JSON, no markdown, no explanation.

<example>
{{"{{jmeno}}": "Jan Novák", "{{datum}}": ""}}
</example>

<context>
{context}
</context>"""
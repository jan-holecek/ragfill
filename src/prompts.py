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
Rewrite the latest question into one or more standalone search queries that are fully
self-contained from the conversation history.
- Resolve pronouns, ellipsis, and references using the conversation history
- Keep each rewritten query a natural, grammatically complete phrase or sentence — do NOT
  strip it down to a bag of bare keywords, natural phrasing embeds better for semantic search
- Keep technical terms, proper nouns, and dates exact
- If answering the question requires combining two or more separate facts (comparing dates,
  computing a difference, joining information about different entities, etc.), split it into
  independent, single-fact sub-questions, one per line — never join them back into a single
  sentence separated by commas or question marks, and never merge them into one line
- Each sub-question must stand completely on its own — if someone read only ONE line without
  the others, it must still be a complete, answerable, self-contained question
- Use 2-3 sub-questions — as many as there are distinct facts to combine, never more than 3
- If the question is already about a single fact, output just one line, as today
- Always output in Czech
- Output ONLY the rewritten query/queries, one per line — no numbering, no bullets, no
  explanation, no extra blank lines between them

Example (single fact, reference resolved from history):
History: "Zajímá mě proces onboardingu." Question: "A kdo za něj zodpovídá?"
-> "Kdo je zodpovědný za proces onboardingu nového zaměstnance?"

Example (single fact, already standalone):
Question: "Kdy se koná příští schůze?"
-> "Kdy se koná příští schůze?"

Example (multi-hop, 2 facts, split into independent sub-questions):
Question: "Kolik let uplynulo mezi odtržením Čech od Velké Moravy a koncem první republiky?"
->
Kdy došlo k odtržení Čech od Velké Moravy?
Kdy skončila první republika?

Example (multi-hop, 3 facts, split into independent sub-questions):
Question: "Kdo vedl Sudetoněmeckou stranu, ve kterém roce Německo získalo sudetská pohraniční území a kolik obyvatel má dnes Ústecký kraj?"
->
Kdo vedl Sudetoněmeckou stranu?
Ve kterém roce Německo získalo sudetská pohraniční území?
Kolik obyvatel má dnes Ústecký kraj?"""


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
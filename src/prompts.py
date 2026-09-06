def build_rag_prompt(context: str) -> str:
    return f"""You are a helpful assistant. Answer the user's question based solely on the provided context.
If the answer is not present in the context, clearly state that you don't know.
Do not make up information that is not in the context.
Always respond in the same language as the user's question.

Context:
{context}"""
def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Build a prompt using the retrieved chunks.
    """

    context = ""
    for chunk in retrieved_chunks:
        context += (
            f"[Source: Page {chunk['page_number']}]\n"
            f"{chunk['text']}\n\n"
        )

    
    prompt = f"""
You are a ServiceNow expert.

Answer the user's question using ONLY the context below.

If the answer is available,
provide a concise answer.

At the end of your answer,
include all page numbers that were used
under a heading called "Sources".

If the answer is not found,
reply:

I don't have enough information.

Context:
{context}

Question:
{query}

Answer:
"""

    return prompt
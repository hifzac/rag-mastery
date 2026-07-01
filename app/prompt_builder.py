def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Build a prompt using the retrieved chunks.
    """

    context = ""

    for chunk in retrieved_chunks:
        context += (
            f"Page {chunk['page_number']}\n"
            f"{chunk['text']}\n\n"
        )

    
    prompt = f"""
You are a ServiceNow expert.

Use ONLY the information in the context below.

If the context contains enough information,
answer the user's question in your own words.

If the answer truly does not exist in the context,
reply exactly:

I don't have enough information.

Context:
{context}

Question:
{query}

Answer:
"""

    return prompt
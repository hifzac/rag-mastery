from typing import List


def chunk_data(
    pages: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> list[dict]:

    chunks = []
    chunk_id = 1

    for page in pages:

        text = page["text"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end]

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "page_number": page["page_number"],
                    "source": page["source"],
                    "text": chunk_text
                }
            )

            chunk_id += 1

            start += chunk_size - chunk_overlap

    return chunks
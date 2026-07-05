from openai import OpenAI
import os
from dotenv import load_dotenv
from config import LLM_MODEL
from memory import get_memory

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


def rewrite_query(query: str, memory: list) -> str:
  

    messages = [
        {
            "role": "system",
            "content": (
                "Rewrite the user's latest question into a standalone "
                "question using the previous conversation if necessary. "
                "Do not answer the question. "
                "Return only the rewritten question."
            )
        }
    ]

    messages.extend(get_memory(memory))

    messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0
    )

    return response.choices[0].message.content.strip()
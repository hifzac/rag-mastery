import os
from dotenv import load_dotenv
from openai import OpenAI
from config import LLM_MODEL
from memory import get_memory
load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


def generate_response(prompt: str, memory: list) -> str:
    """
    Generate an answer using the current prompt and chat history.
    """

    messages = [
        {
            "role": "system",
            "content": (
                "You are a ServiceNow expert. "
                "Answer only from the provided context."
            )
        }
    ]

    # Add previous conversation
    messages.extend(get_memory(memory))

    # Add current prompt
    messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.2,
        stream=True
    )

    for chunk in stream:

        if (
            chunk.choices
            and chunk.choices[0].delta.content is not None
        ):
            yield chunk.choices[0].delta.content
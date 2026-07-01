MAX_MEMORY = 10  


def add_to_memory(memory: list, role: str, content: str):

    memory.append(
        {
            "role": role,
            "content": content
        }
    )

    if len(memory) > MAX_MEMORY:
        memory.pop(0)


def get_memory(memory: list):
    """
    Return conversation history.
    """
    return memory
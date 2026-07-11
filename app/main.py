from packaging import metadata

from loader import load_pdf
from chunker import chunk_data
from embeddings import generate_embeddings
from vector_store import (
    build_index,
    save_index,
    load_index,
)
from retriever import retrieve
from prompt_builder import build_prompt
from llm import generate_response
from memory import add_to_memory

from config import (
    BM25_PATH,
    PDF_PATH,
    INDEX_PATH,
    METADATA_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVAL_TOP_K,
    RERANK_TOP_K,
)
from bm25_store import (
    build_bm25_index,
    save_bm25
)

def main():

    # -----------------------------------------
    # Build Index (Only First Time)
    # -----------------------------------------

    if (
    not INDEX_PATH.exists()
    or not METADATA_PATH.exists()
    or not BM25_PATH.exists()
    or BM25_PATH.stat().st_size == 0
    ):

        print("Loading PDF...")

        pages = load_pdf(PDF_PATH)

        print(f"Loaded {len(pages)} pages.")

        print("Creating Chunks...")

        chunks = chunk_data(
            pages,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        print(f"Created {len(chunks)} chunks.")

        print("Generating Embeddings...")

        embedded_chunks = generate_embeddings(chunks)

        print("Building FAISS Index...")

        index, metadata = build_index(embedded_chunks)
        bm25 = build_bm25_index(metadata)
        save_index(
            index,
            metadata,
            INDEX_PATH,
            METADATA_PATH,
        )
        save_bm25(
        bm25,
        BM25_PATH
        )
        print("Index Created Successfully!\n")

    # -----------------------------------------
    # Load Existing Index
    # -----------------------------------------

    print("Loading FAISS Index...")

    index, metadata = load_index(
        INDEX_PATH,
        METADATA_PATH,
    )
    
    print(f"Loaded {index.ntotal} vectors.\n")

    # -----------------------------------------
    # Conversation Memory
    # -----------------------------------------

    chat_memory = []

    # -----------------------------------------
    # Chat Loop
    # -----------------------------------------

    while True:

        query = input("\nAsk a question (type 'exit' to quit): ")

        if query.lower() == "exit":
            print("Goodbye!")
            break

        # Retrieve relevant chunks
        results = retrieve(
            query=query,
            index=index,
            metadata=metadata,
            top_k=RETRIEVAL_TOP_K,
        )

        # Build prompt
        prompt = build_prompt(
            query=query,
            retrieved_chunks=results,
        )

        # Generate answer (uses memory)
        answer = generate_response(
            prompt=prompt,
            memory=chat_memory,
        )

        # Save conversation
        add_to_memory(
            chat_memory,
            role="user",
            content=query,
        )

        add_to_memory(
            chat_memory,
            role="assistant",
            content=answer,
        )

        # Print answer
        print("\nAnswer")
        print("=" * 80)
        print(answer)

        # Print sources
        pages = sorted(
            {
                chunk["page_number"]
                for chunk in results
            }
        )

        print("\nSources")
        print("=" * 80)

        for page in pages:
            print(f"Page {page}")

        print("=" * 80)


if __name__ == "__main__":
    main()
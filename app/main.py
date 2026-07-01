from pathlib import Path

from loader import load_pdf
from chunker import chunk_data
from embeddings import generate_embeddings
from vector_store import (
    build_index,
    save_index,
    load_index
)
from retriever import retrieve
from prompt_builder import build_prompt
from llm import generate_response

def main():

    # -------------------------------
    # Project Root
    # -------------------------------
    project_root = Path(__file__).resolve().parent.parent

    # -------------------------------
    # File Paths
    # -------------------------------
    pdf_path = (
        project_root
        / "data"
        / "servicenow-australia-operational-technology-enus.pdf"
    )

    index_path = project_root / "data" / "faiss_index.bin"
    metadata_path = project_root / "data" / "metadata.pkl"

    # -------------------------------
    # Build Index (Run Only Once)
    # -------------------------------
    if not index_path.exists():

        print("Loading PDF...")
        pages = load_pdf(pdf_path)
        print(f"Loaded {len(pages)} pages.")

        print("Creating chunks...")
        chunks = chunk_data(
            pages,
            chunk_size=1000,
            chunk_overlap=200
        )
        print(f"Created {len(chunks)} chunks.")

        print("Generating embeddings...")
        embedded_chunks = generate_embeddings(chunks)

        print("Building FAISS index...")
        index, metadata = build_index(embedded_chunks)

        print("Saving index...")
        save_index(
            index,
            metadata,
            index_path,
            metadata_path
        )

        print("Index saved successfully!\n")

    # -------------------------------
    # Load Existing Index
    # -------------------------------
    print("Loading FAISS index...")

    index, metadata = load_index(
        index_path,
        metadata_path
    )

    print(f"Index Loaded ({index.ntotal} vectors)\n")

    # -------------------------------
    # Ask Questions
    # -------------------------------
    while True:

        query = input("Ask a question (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        # Retrieve relevant chunks
        results = retrieve(
            query=query,
            index=index,
            metadata=metadata,
            top_k=3
        )

        # Print retrieved chunks (for debugging)
        print("\nTop Relevant Chunks")
        print("=" * 80)

        for i, chunk in enumerate(results, start=1):
            print(f"Result {i}")
            print(f"Chunk ID    : {chunk['chunk_id']}")
            print(f"Page Number : {chunk['page_number']}")
            print("-" * 80)
            print(chunk["text"][:300])
            print()

        # Build Prompt
        prompt = build_prompt(query, results)
        answer = generate_response(prompt)
        

        pages = sorted(
            set(chunk["page_number"] for chunk in results)
        )

        print(answer)

        print("\nSources:")

        for page in pages:
            print(f"- Page {page}")
       
       


if __name__ == "__main__":
    main()
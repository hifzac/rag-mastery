from pathlib import Path
from loader import load_pdf
from chunker import chunk_data

def main():
    project_root = Path(__file__).resolve().parent.parent

    pdf_path = project_root / "data" / "servicenow-australia-operational-technology-enus.pdf"

    print(f"Reading PDF from: {pdf_path}")

    pages = load_pdf(pdf_path)
    chunks = chunk_data(pages, chunk_size=1000, chunk_overlap=200)
    print(f"Total Pages: {len(pages)}")
    print(f"Total Chunks: {len(chunks)}")

    print("\nFirst Page:\n")
    print(pages[0]["text"][:1000])
    


if __name__ == "__main__":
    main()
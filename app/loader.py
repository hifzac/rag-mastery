from pathlib import Path
from pypdf import PdfReader

def load_pdf(file_path: Path)->list[dict]:
    pdf_path = Path(file_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")
    try:
        reader=PdfReader(str(pdf_path))
        pages=[]
        for page_number,page in enumerate(reader.pages, start=1):
             text = page.extract_text()
             pages.append(
                 {
                     "page_number":page_number,
                     "text":text if text else "",
                      "source": pdf_path.name
                 }
             )
        return pages

    except Exception as e:
        raise Exception(f"Error reading PDF: {e}")

           

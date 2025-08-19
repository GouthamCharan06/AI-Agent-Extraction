import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.ingestion import extract_text_from_pdf
from core.routing import detect_doc_type

if __name__ == "__main__":
    sample_path = os.path.join("data", "Invoice-KS  -2034.pdf")

    text = extract_text_from_pdf(sample_path)
    print("Extracted text preview:", text[:300])

    doc_type = detect_doc_type(text)
    print("Document classified as:", doc_type)

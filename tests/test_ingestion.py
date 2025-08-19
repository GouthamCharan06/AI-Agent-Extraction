print("🚀 Test script started")
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
from core.ingestion import extract_text_from_pdf

if __name__ == "__main__":
    # Path to your sample PDF
    sample_path = os.path.join("data", "Invoice-KS  -2034.pdf")

    if not os.path.exists(sample_path):
        print("Sample file not found, please add a PDF in data/ folder")
    else:
        text = extract_text_from_pdf(sample_path)
        print("Extraction complete, text preview:\n")
        print(text[:1000])  # print first 1000 characters only


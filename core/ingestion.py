import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
import easyocr

# Set tesseract path (Windows only, not used in Streamlit Cloud)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF. First try PyMuPDF, fallback to OCR."""
    text = ""

    # 1. Direct text extraction (digital PDFs)
    doc = fitz.open(file_path)
    for page in doc:
        page_text = page.get_text()
        if page_text:
            text += page_text + "\n"
    doc.close()

    if text.strip():
        return text.strip()

    # 2. Fallback: Scanned PDF -> OCR
    images = convert_from_path(file_path)

    try:
        # Try pytesseract
        text = " ".join([pytesseract.image_to_string(img) for img in images])
    except Exception:
        # Fallback: EasyOCR (works on Streamlit Cloud)
        reader = easyocr.Reader(["en"])
        text = " ".join(
            [" ".join(result[1] for result in reader.readtext(img)) for img in images]
        )

    return text.strip()

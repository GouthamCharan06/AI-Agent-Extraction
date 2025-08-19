import fitz  # PyMuPDF
from pdf2image import convert_from_path
import easyocr
from PIL import Image
import os

# Initialize EasyOCR globally (Streamlit Cloud compatible)
OCR_READER = easyocr.Reader(["en"], gpu=False)

def extract_text_from_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in [".png", ".jpg", ".jpeg"]:
        return extract_text_from_image(file_path)
    else:
        print(f"[Warning] Unsupported file type: {ext}")
        return ""

def extract_text_from_pdf(file_path: str) -> str:
    text = extract_text_digital(file_path)
    ocr_text = extract_text_ocr(file_path)

    if text.strip():
    # Combine both text and ocr for safety
        return (text + "\n" + ocr_text).strip()
    else:
        return ocr_text.strip()

  

def extract_text_digital(file_path: str) -> str:
    try:
        doc = fitz.open(file_path)
        text = "\n".join([page.get_text() for page in doc])
        doc.close()
        return text
    except Exception as e:
        print(f"[Warning] Digital PDF extraction failed: {e}")
        return ""

def extract_text_ocr(file_path: str) -> str:
    try:
        images = convert_from_path(file_path)
    except Exception as e:
        print(f"[Error] Failed to convert PDF to images: {e}")
        return ""
    all_text = []
    for img in images:
        try:
            result = OCR_READER.readtext(img)
            page_text = " ".join([res[1] for res in result])
            all_text.append(page_text)
        except Exception as e:
            print(f"[Error] EasyOCR failed for PDF page: {e}")
    return "\n".join(all_text)

def extract_text_from_image(file_path: str) -> str:
    try:
        img = Image.open(file_path)
        result = OCR_READER.readtext(img)
        text = " ".join([res[1] for res in result])
        return text.strip()
    except Exception as e:
        print(f"[Error] EasyOCR failed for image {file_path}: {e}")
        return ""

import fitz  # PyMuPDF
import easyocr
from PIL import Image
import cv2
import numpy as np
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
    digital_text = extract_text_digital(file_path)
    ocr_text = extract_text_ocr(file_path)
    
    # Combine both outputs
    combined_text = "\n".join([digital_text, ocr_text]).strip()
    return combined_text if combined_text else ""

def extract_text_digital(file_path: str) -> str:
    try:
        doc = fitz.open(file_path)
        text = "\n".join([page.get_text() for page in doc])
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"[Warning] Digital PDF extraction failed: {e}")
        return ""

def extract_text_ocr(file_path: str) -> str:
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        print(f"[Error] Failed to open PDF for OCR: {e}")
        return ""

    all_text = []
    for page_num, page in enumerate(doc):
        try:
            pix = page.get_pixmap()  # Render page as image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            # Convert to OpenCV grayscale and threshold
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            _, img_bin = cv2.threshold(img_cv, 150, 255, cv2.THRESH_BINARY)
            # OCR
            result = OCR_READER.readtext(img_bin)
            page_text = " ".join([res[1] for res in result])
            all_text.append(page_text)
        except Exception as e:
            print(f"[Error] EasyOCR failed for PDF page {page_num + 1}: {e}")
    return "\n".join(all_text)

def extract_text_from_image(file_path: str) -> str:
    try:
        img = Image.open(file_path)
        # Convert to OpenCV grayscale and threshold
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        _, img_bin = cv2.threshold(img_cv, 150, 255, cv2.THRESH_BINARY)
        result = OCR_READER.readtext(img_bin)
        text = " ".join([res[1] for res in result])
        return text.strip()
    except Exception as e:
        print(f"[Error] EasyOCR failed for image {file_path}: {e}")
        return ""

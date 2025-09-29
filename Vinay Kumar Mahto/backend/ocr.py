import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to find tesseract in common Linux paths
TESSERACT_PATHS = [
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
    os.environ.get("TESSERACT_PATH", "tesseract")  # fallback to PATH
]

tesseract_found = False
for path in TESSERACT_PATHS:
    if path and os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        tesseract_found = True
        logger.info(f"Tesseract found at {path}")
        break

if not tesseract_found:
    logger.info("Using tesseract from PATH (default behavior)")

def preprocess_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    _, thresh = cv2.threshold(denoised, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(thresh)

def extract_text_from_scanned_pdf(pdf_path, dpi=300):
    try:
        images = convert_from_path(pdf_path, dpi=dpi)
        ocr_text = ""
        ocr_data_pages = []
        for img in images:
            processed = preprocess_image(img)
            text = pytesseract.image_to_string(processed, lang="eng")
            ocr_data = pytesseract.image_to_data(processed, lang="eng", output_type=pytesseract.Output.DICT)
            ocr_text += text + "\n"
            ocr_data_pages.append((img, ocr_data))
        return True, ocr_text, ocr_data_pages
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        return False, "", []



print(cv2.__version__)

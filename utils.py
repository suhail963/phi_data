import cv2
import pytesseract
import numpy as np
from pdf2image import convert_from_bytes

# Set Tesseract command path if not in PATH
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Image preprocessing function
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# OCR text extraction function
def extract_text_lines(image):
    config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(image, config=config)
    return [line.strip() for line in text.splitlines() if line.strip()]

# OCR data extraction with pandas dataframe
def extract_ocr_data(image):
    config = r'--oem 3 --psm 6'
    ocr_data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DATAFRAME)
    print("[DEBUG] OCR Data:")
    print(ocr_data.head())  # Debug the first few rows to check for duplicates
    return ocr_data

# Convert PDF to images
def convert_pdf_to_images(file_bytes):
    return convert_from_bytes(file_bytes, dpi=300)

# Example main process
def process_hybrid(file_bytes, file_ext):
    if file_ext == ".pdf":
        images = convert_pdf_to_images(file_bytes)
        for img in images:
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            pre = preprocess_image(img_cv)
            lines = extract_text_lines(pre)
            ocr_data = extract_ocr_data(pre)  # Get structured OCR data
            
            # Example: Merge extracted data into DataFrame and handle duplicates
            print("[DEBUG] OCR Data processed:")
            print(ocr_data.head())  # View the processed OCR data

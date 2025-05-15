import sys
import os
import pandas as pd
import cv2
import numpy as np
from utils import preprocess_image, extract_text_lines, convert_pdf_to_images
from flow2_targeted import match_fields
from phi.llm.google import Gemini
from phi.llm.message import Message
from dotenv import load_dotenv
import json
import re

# Load the .env file
load_dotenv()

# Fetch the API key from the environment
api_key = os.getenv('GOOGLE_API_KEY')

if not api_key:
    print("[!] API Key not found in .env file!")
    sys.exit(1)

# Initialize the Gemini agent with the API key
ocr_agent = Gemini(api_key=api_key)

def generate_prompt(fields, document_text):
    field_list = ", ".join(fields)
    prompt = (
        f"Given the OCR-extracted text of a document, extract only the following fields: {field_list}.\n\n"
        "Please return the extracted fields in the following format:\n"
        "{\n"
        "  \"Field Name\": \"Value\",\n"
        "  \"Field Name 2\": \"Value\",\n"
        "  ...\n"
        "}\n\n"
        "Do not include any extra explanations, formatting, or content beyond the JSON response.\n\n"
        f"Document text:\n{document_text}"
    )
    return prompt

def process_hybrid(file_bytes, file_ext, fields, output_csv="output.csv"):
    if not fields:
        print("[!] No fields provided. Exiting.")
        return

    lines = []

    # Convert PDF or image to text lines
    if file_ext == ".pdf":
        images = convert_pdf_to_images(file_bytes)
        for img in images:
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            pre = preprocess_image(img_cv)
            lines += extract_text_lines(pre)
    else:
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        pre = preprocess_image(img)
        lines = extract_text_lines(pre)

    document_text = "\n".join(lines)

    # Step 1: Deterministic extraction of required fields
    required_data = match_fields(lines, fields)

    # Ensure required_data is a dictionary
    if not isinstance(required_data, dict):
        print(f"[!] Expected a dictionary for required_data, but got {type(required_data)}")
        return

    # Step 2: NLP-based inferred field extraction using Gemini
    try:
        full_prompt = generate_prompt(fields, document_text)
        messages = [Message(role="user", parts=[full_prompt])]
        response = ocr_agent.invoke(messages)

        print("[DEBUG] Gemini raw response:")
        print(repr(response.text))

        if not response or not response.text.strip():
            raise ValueError("Empty response from Gemini")

        # Clean Markdown code block (```json ... ```)
        cleaned_text = re.sub(r'^```(?:json)?\s*|```$', '', response.text.strip(), flags=re.MULTILINE).strip()

        print("[DEBUG] Cleaned Gemini response:")
        print(cleaned_text)

        inferred_fields = json.loads(cleaned_text)

    except Exception as e:
        print(f"[!] Error in Gemini response: {e}")
        inferred_fields = [{"Field": "Error", "Value": str(e), "Confidence": "0"}]

    # Check if the response is a dictionary, and wrap it in a list if so
    if isinstance(inferred_fields, dict):
        inferred_fields = [inferred_fields]

    # Ensure inferred_fields is a list of dictionaries
    if not isinstance(inferred_fields, list):
        print(f"[!] Expected a list of dictionaries for inferred_fields, but got {type(inferred_fields)}")
        return

    # Step 3: Merge both sets of data (required fields and inferred fields)
    final_data = []
    for item in inferred_fields:
        if isinstance(item, dict):  # Ensure item is a dictionary
            row = {**required_data, **item}
            final_data.append(row)
        else:
            print(f"[!] Skipping non-dictionary item: {item}")

    df = pd.DataFrame(final_data)

    print("[DEBUG] DataFrame before saving to CSV:")
    print(df)

    try:
        df.to_csv(output_csv, index=False)
        print(f"[✓] Output saved to {output_csv}")
    except Exception as e:
        print(f"[!] Failed to save CSV: {e}")

# Command-line argument processing
if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python flow4_hybrid.py <file_path> <fields> <prompt> <output_csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    fields = sys.argv[2].split(",")  # Split the fields provided by user
    prompt = sys.argv[3]  # Prompt not currently used dynamically but can be if necessary
    output_csv = sys.argv[4]

    # Ensure the file exists
    if not os.path.exists(file_path):
        print(f"[!] File not found: {file_path}")
        sys.exit(1)

    # Get file extension
    file_ext = os.path.splitext(file_path)[-1].lower()
    if file_ext not in [".png", ".jpg", ".jpeg", ".pdf"]:
        print("[!] Only PDF or image files are supported.")
        sys.exit(1)

    # Read the file as bytes
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    # Process the document and save the result
    process_hybrid(file_bytes, file_ext, fields, output_csv)

import pandas as pd
import re
from utils import preprocess_image, extract_text_lines, convert_pdf_to_images
import cv2
import numpy as np
import json
import sys
import os

def match_fields(text_lines, target_fields):
    result = {}
    for field in target_fields:
        found = "NA"
        for line in text_lines:
            # Normalize case
            if re.search(re.escape(field), line, re.IGNORECASE):
                # Try to extract the numeric or dollar amount from the line
                match = re.search(r'[\d\.,]+', line)
                found = match.group(0) if match else line.strip()
                break
        result[field] = found
    return result

def process_targeted_fields(file_bytes, file_ext, fields_json):
    target_fields = json.loads(fields_json)
    lines = []

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

    data = match_fields(lines, target_fields)
    return pd.DataFrame([data])

# === Entry point for running from command line ===
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python flow2_targeted.py <input_file> <field1> <field2> ... <output_csv>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[-1]
    field_args = sys.argv[2:-1]  # all fields between input and output

    file_ext = os.path.splitext(input_path)[-1].lower()

    with open(input_path, "rb") as f:
        file_bytes = f.read()

    # Convert field arguments to JSON string for internal processing
    fields_json = json.dumps(field_args)
    df_result = process_targeted_fields(file_bytes, file_ext, fields_json)
    df_result.to_csv(output_path, index=False)
    print(f"Saved output to {output_path}")

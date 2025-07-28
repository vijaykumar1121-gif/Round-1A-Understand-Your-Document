import os
import fitz  # PyMuPDF
import pandas as pd
import joblib
import json
from statistics import mode
import unicodedata

# Local input/output directories
INPUT_DIR = "input"   # Change if needed
OUTPUT_DIR = "output" # Change if needed

# Load your model
model = joblib.load("heading_classifier_lgbm2.pkl")  # Make sure this file is present

def rgb_int_to_hex(rgb_int):
    r = (rgb_int >> 16) & 255
    g = (rgb_int >> 8) & 255
    b = rgb_int & 255
    return "#{:02X}{:02X}{:02X}".format(r, g, b)

def detect_script(text):
    for ch in text:
        name = unicodedata.name(ch, "")
        if "CJK UNIFIED" in name or "HIRAGANA" in name or "KATAKANA" in name:
            return "japanese"
    return "latin"

def extract_features(pdf_path):
    doc = fitz.open(pdf_path)
    dataset = []
    font_sizes = []

    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    font_sizes.append(span["size"])

    if not font_sizes:
        return []

    body_size = mode(font_sizes)

    for page_num, page in enumerate(doc, start=1):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                line_text = ""
                sizes = []
                fonts = []
                bold_flag = False
                color = None
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    line_text += text + " "
                    sizes.append(span["size"])
                    fonts.append(span["font"])
                    if "Bold" in span["font"]:
                        bold_flag = True
                    color = span["color"]

                line_text = line_text.strip()
                if not line_text:
                    continue

                avg_size = round(sum(sizes) / len(sizes), 2)
                most_common_font = max(set(fonts), key=fonts.count)
                script = detect_script(line_text)
                is_caps = 1 if script == "latin" and line_text.isupper() else 0

                if avg_size <= body_size:
                    continue

                dataset.append({
                    "text": line_text,
                    "font_size": avg_size,
                    "font_name": most_common_font,
                    "is_bold": int(bold_flag),
                    "is_caps": is_caps,
                    "page_number": page_num,
                    "line_length": len(line_text),
                    "color_rgb": rgb_int_to_hex(color),
                    "script_type": script
                })

    return dataset

def predict_headings(data):
    df = pd.DataFrame(data)

    df[["r", "g", "b"]] = df["color_rgb"].str.extract(r"#(..)(..)(..)")
    df["r"] = df["r"].map(lambda x: int(x, 16) if pd.notna(x) else 0)
    df["g"] = df["g"].map(lambda x: int(x, 16) if pd.notna(x) else 0)
    df["b"] = df["b"].map(lambda x: int(x, 16) if pd.notna(x) else 0)
    df.drop(columns=["color_rgb"], inplace=True)

    df = pd.get_dummies(df, columns=["font_name", "script_type"], prefix=["font", "lang"])
    df.columns = df.columns.str.replace(r"[^\w]", "_", regex=True)

    for col in model.feature_name_:
        if col not in df.columns:
            df[col] = 0
    df = df[model.feature_name_]

    predictions = model.predict(df)
    result = []
    for i in range(len(predictions)):
        entry = data[i]
        entry["label"] = predictions[i]
        result.append(entry)
    return result

def process_pdf(pdf_path, output_path):
    extracted = extract_features(pdf_path)
    if not extracted:
        return

    result = predict_headings(extracted)

    # Apply stricter filtering for H3 (only lines with exactly 6 words)
    filtered = []
    for r in result:
        if r["label"] in ("H1", "H2"):
            filtered.append(r)
        elif r["label"] == "H3":
            word_count = len(r["text"].split())
            if word_count == 6:
                filtered.append(r)

    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    output = {
        "title": filename,
        "outline": [
            {
                "level": item["label"],
                "text": item["text"],
                "page": item["page_number"]
            }
            for item in filtered
        ]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for fname in os.listdir(INPUT_DIR):
        if fname.lower().endswith(".pdf"):
            in_file = os.path.join(INPUT_DIR, fname)
            out_file = os.path.join(OUTPUT_DIR, os.path.splitext(fname)[0] + ".json")
            process_pdf(in_file, out_file)

if __name__ == "__main__":
    main()

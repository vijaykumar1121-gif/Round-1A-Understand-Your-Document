# 📄 Round 1A: Understand Your Document

## 🚀 Overview

This solution processes a given PDF to extract a structured outline that includes the document **title** and its headings: **H1**, **H2**, and **H3** (with level and page number). It is designed to work offline, within 10 seconds for a 50-page PDF, and can be containerized for consistent and reproducible execution.

---

## 🧠 What It Does

The system:
- Automatically processes all PDFs in the `/app/input` folder.
- Uses a LightGBM-based ML model to classify PDF lines into headings (H1/H2/H3).
- Applies specific filters.
- Outputs structured JSON files in `/app/output`, with one `.json` for each `.pdf`.

---

## 🗂 Input/Output Example

### ✅ Input: PDF (placed inside `/app/input/`)
Supports English, Japanese, and mixed-script PDFs.

### ✅ Output: JSON (saved inside `/app/output/`)

```json
{
  "title": "Understanding AI",
  "outline": [
    { "level": "H1", "text": "Introduction", "page": 1 },
    { "level": "H2", "text": "What is AI?", "page": 2 },
    { "level": "H3", "text": "History of AI", "page": 3 }
  ]
}
```

---

## 🛠️ How It Works

The model uses features such as:
- Positioning of Words
- Font size and boldness
- Capitalization (for Latin script)
- Font name and color
- Script type (Latin or Japanese)
- Sentence length.

It extracts this data using **PyMuPDF** (`fitz`) and feeds it into a LightGBM classifier.

---

## ⚙️ How to Run

### 🐳 Docker Execution

**Build the Docker image:**

```bash
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .
```

**Run the Docker container:**

```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  mysolutionname:somerandomidentifier
```

### 🐍 Normal Python Execution (No Docker)

1. Install requirements:

```bash
pip install -r requirements.txt
```

2. Place your PDFs in the `input/` directory.

3. Run the script:

```bash
python main.py
```

4. Extracted JSON files will be saved in the `output/` directory.

---

## 🧾 Expected Behavior

For every file `/app/input/<filename>.pdf`, it generates `/app/output/<filename>.json`.

---

## 📦 Dependencies

All dependencies are listed in `requirements.txt`:

- Python 3.10
- PyMuPDF
- pandas
- joblib
- lightgbm
- scikit-learn

---

## 🌍 Real-World Applications

This project is useful for building intelligent systems that understand the structure of documents. Real-world use cases include:

- 📚 **Semantic search engines**: Enable users to search document sections based on headings.
- 🧠 **AI summarization tools**: Use extracted outlines to create summaries and navigation.
- 🗂️ **Content classification and tagging**: Automatically categorize document sections for better indexing.
- 🔍 **Document comparison**: Align sections between versions using structured outlines.
- 📊 **Data analytics dashboards**: Extract structured data from business reports, policy PDFs, research papers, etc.

---

## 📌 Constraints Compliance

| Constraint               | Status ✅        |
|-------------------------|------------------|
| ⏱ Execution time        | ≤ 10s for 50 pages |
| 📦 Model size            | ≤ 200MB           |
| 🌐 Network               | No internet used  |
| ⚙️ Runtime               | CPU (amd64)  |
| 📁 I/O Format            | Per PDF → Per JSON |

---

## ✅ Submission Checklist

- [x] Git project with working Dockerfile
- [x] All dependencies installed inside container
- [x] Output matches required JSON format
- [x] Docker-compatible execution (no internet, CPU-only)


# Textract Table OCR (Tables only)

A minimal local CLI to extract tables using Amazon AWS Textract and save them as CSV or Excel files.

- Images (PNG/JPG/TIFF) use the synchronous Textract `analyze_document` API.
- PDFs use the asynchronous Textract `start_document_analysis` API and require an S3 bucket for upload.
- No pre or post processing is performed. Raw Textract table extraction only.

## Prerequisites
- Python 3.10+
- AWS credentials configured locally (e.g., environment variables, `~/.aws/credentials`, or an instance profile)
- An S3 bucket (only needed when processing PDFs)

Textract charges apply. Ensure you understand AWS costs before running.

## Installation
```bash
# Create a virtual environment (recommended)
python3 -m venv .venv

# Upgrade pip and install deps
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

## Usage
Run the CLI via `-m`:

```bash
# Help
.venv/bin/python -m textract_table_ocr.cli --help
```

### Images (PNG/JPG/JPEG/TIFF)
```bash
.venv/bin/python -m textract_table_ocr.cli \
  --input path/to/image.jpg \
  --output output.xlsx
```

### PDFs (requires S3 bucket)
Textract requires PDFs to be read from S3 for async processing.
```bash
.venv/bin/python -m textract_table_ocr.cli \
  --input path/to/doc.pdf \
  --s3-bucket your-bucket-name \
  --s3-prefix optional/prefix/ \
  --output output.xlsx
```

### Output format
- If `--output` ends with `.xlsx`, all tables are written to a single Excel workbook (one sheet per table).
- If `--output` ends with `.csv`:
  - One table -> a single CSV file.
  - Multiple tables -> multiple CSV files: `<base>_table1.csv`, `<base>_table2.csv`, etc.
- If `--output` is omitted, defaults to `<input_stem>_tables.xlsx` in the current directory.

## Notes & Limitations
- Tables are reconstructed using Textract `CELL` metadata. Row/column spans are recorded but not expanded; text is placed in the top-left spanned cell.
- No image pre-processing, post-processing, or table cleanup is performed.
- For PDFs, you must supply `--s3-bucket`. The file will be uploaded before starting the Textract job.
- Use `--region` to override the AWS region if not set in your environment.

## Example
```bash
# Image to Excel
.venv/bin/python -m textract_table_ocr.cli -i sample.jpg -o sample_tables.xlsx

# PDF to CSVs (multiple tables)
.venv/bin/python -m textract_table_ocr.cli -i sample.pdf \
  --s3-bucket my-textract-inputs \
  -o sample_tables.csv
```

## Project Structure
```
textract_table_ocr/
  ├─ textract_table_ocr/
  │   ├─ __init__.py
  │   ├─ cli.py
  │   └─ textract_tables.py
  ├─ requirements.txt
  └─ README.md
```

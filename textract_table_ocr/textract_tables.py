from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
import pandas as pd


def _is_pdf(path: Path) -> bool:
    return path.suffix.lower() == ".pdf"


def _compose_s3_key(prefix: str, file_name: str) -> str:
    if not prefix:
        return file_name
    if not prefix.endswith("/"):
        prefix = prefix + "/"
    return prefix + file_name


def _session(region: Optional[str] = None, profile: Optional[str] = None) -> boto3.session.Session:
    return boto3.session.Session(region_name=region, profile_name=profile)


def _get_text_for_block(block: dict, block_map: Dict[str, dict]) -> str:
    pieces: List[str] = []
    for rel in block.get("Relationships", []) or []:
        if rel.get("Type") == "CHILD":
            for cid in rel.get("Ids", []) or []:
                child = block_map.get(cid)
                if not child:
                    continue
                btype = child.get("BlockType")
                if btype == "WORD":
                    t = child.get("Text", "")
                    if t:
                        pieces.append(t)
                elif btype == "SELECTION_ELEMENT":
                    if child.get("SelectionStatus") == "SELECTED":
                        pieces.append("X")
    return " ".join(pieces).strip()


def _extract_tables_from_blocks(blocks: List[dict]) -> List[pd.DataFrame]:
    block_map: Dict[str, dict] = {b["Id"]: b for b in blocks if "Id" in b}
    tables: List[pd.DataFrame] = []

    for b in blocks:
        if b.get("BlockType") != "TABLE":
            continue

        cell_ids: List[str] = []
        for rel in b.get("Relationships", []) or []:
            if rel.get("Type") == "CHILD":
                for cid in rel.get("Ids", []) or []:
                    cb = block_map.get(cid)
                    if cb and cb.get("BlockType") == "CELL":
                        cell_ids.append(cid)

        # Map cells by (row, col)
        cells: Dict[Tuple[int, int], str] = {}
        max_row = 0
        max_col = 0
        for cid in cell_ids:
            cell = block_map.get(cid)
            if not cell:
                continue
            row = int(cell.get("RowIndex", 0))
            col = int(cell.get("ColumnIndex", 0))
            max_row = max(max_row, row)
            max_col = max(max_col, col)
            text = _get_text_for_block(cell, block_map)
            # Place text in the top-left of spanned area
            cells[(row, col)] = text

        # Build 2D array
        grid: List[List[str]] = [["" for _ in range(max_col)] for _ in range(max_row)]
        for (r, c), txt in cells.items():
            # Convert 1-based to 0-based
            if r >= 1 and c >= 1:
                grid[r - 1][c - 1] = txt

        tables.append(pd.DataFrame(grid))

    return tables


def _analyze_image_tables(sess: boto3.session.Session, input_path: Path) -> List[pd.DataFrame]:
    client = sess.client("textract")
    data = input_path.read_bytes()
    resp = client.analyze_document(Document={"Bytes": data}, FeatureTypes=["TABLES"])  # type: ignore[call-arg]
    blocks = resp.get("Blocks", []) or []
    return _extract_tables_from_blocks(blocks)


def _analyze_pdf_tables(sess: boto3.session.Session, input_path: Path, s3_bucket: str, s3_prefix: str = "") -> List[pd.DataFrame]:
    # Upload to S3
    s3 = sess.client("s3")
    key = _compose_s3_key(s3_prefix, input_path.name)
    s3.upload_file(str(input_path), s3_bucket, key)

    client = sess.client("textract")
    start = client.start_document_analysis(
        DocumentLocation={"S3Object": {"Bucket": s3_bucket, "Name": key}},
        FeatureTypes=["TABLES"],
    )
    job_id = start["JobId"]

    # Poll for completion
    while True:
        resp = client.get_document_analysis(JobId=job_id, MaxResults=1000)
        status = resp.get("JobStatus")
        if status == "SUCCEEDED" or status == "PARTIAL_SUCCESS":
            all_blocks: List[dict] = []
            next_token = None
            # Accumulate pages
            while True:
                if next_token:
                    resp = client.get_document_analysis(JobId=job_id, NextToken=next_token, MaxResults=1000)
                all_blocks.extend(resp.get("Blocks", []) or [])
                next_token = resp.get("NextToken")
                if not next_token:
                    break
            return _extract_tables_from_blocks(all_blocks)
        if status == "FAILED":
            raise RuntimeError(f"Textract job failed for {input_path} (JobId={job_id})")
        time.sleep(2.0)


def extract_tables_from_file(
    input_path: Path | str,
    *,
    region: Optional[str] = None,
    profile: Optional[str] = None,
    s3_bucket: Optional[str] = None,
    s3_prefix: str = "",
) -> List[pd.DataFrame]:
    """Extract tables from an input file (image or PDF) using AWS Textract.

    - Images are processed synchronously via AnalyzeDocument.
    - PDFs are processed asynchronously via StartDocumentAnalysis and require S3.
    """
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(p)

    sess = _session(region=region, profile=profile)

    if _is_pdf(p):
        if not s3_bucket:
            raise ValueError("s3_bucket is required for PDF inputs")
        return _analyze_pdf_tables(sess, p, s3_bucket=s3_bucket, s3_prefix=s3_prefix)
    else:
        return _analyze_image_tables(sess, p)


def default_output_path(input_path: Path | str) -> Path:
    p = Path(input_path)
    return p.with_name(p.stem + "_tables.xlsx")


def save_tables(tables: List[pd.DataFrame], output_path: Path | str) -> List[Path]:
    """Save tables to CSV or Excel. Returns the list of created file paths.

    - .xlsx: one workbook, one sheet per table
    - .csv: one file if 1 table, else multiple files suffixed with _tableN
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Always have at least one table to emit
    if not tables:
        tables = [pd.DataFrame()]

    ext = out.suffix.lower()
    created: List[Path] = []

    if ext in {".xlsx", ".xls"}:
        # Normalize to .xlsx if .xls was given
        if ext == ".xls":
            out = out.with_suffix(".xlsx")
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            for i, df in enumerate(tables, start=1):
                sheet = f"Table{i}"
                # Write data without headers/index to keep raw table grid
                df.to_excel(writer, sheet_name=sheet, index=False, header=False)
        created.append(out)
        return created

    if ext == ".csv":
        if len(tables) == 1:
            tables[0].to_csv(out, index=False, header=False)
            created.append(out)
        else:
            base = out.with_suffix("")
            for i, df in enumerate(tables, start=1):
                f = base.parent / f"{base.name}_table{i}.csv"
                df.to_csv(f, index=False, header=False)
                created.append(f)
        return created

    # Default to .xlsx if no/unknown extension
    out = out.with_suffix(".xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for i, df in enumerate(tables, start=1):
            sheet = f"Table{i}"
            df.to_excel(writer, sheet_name=sheet, index=False, header=False)
    created.append(out)
    return created

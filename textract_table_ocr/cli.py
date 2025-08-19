from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .textract_tables import (
    default_output_path,
    extract_tables_from_file,
    save_tables,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract tables using AWS Textract and save to CSV/Excel.")
    p.add_argument("-i", "--input", required=True, help="Path to input image or PDF")
    p.add_argument("-o", "--output", help="Output file path (.xlsx or .csv). Defaults to <input>_tables.xlsx")
    p.add_argument("--s3-bucket", help="S3 bucket for PDF processing (required for PDFs)")
    p.add_argument("--s3-prefix", default="", help="Optional S3 prefix for uploaded PDF")
    p.add_argument("--region", help="AWS region (overrides environment)")
    p.add_argument("--profile", help="AWS profile name (optional)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)

    out_path: Path
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = default_output_path(in_path)

    tables = extract_tables_from_file(
        in_path,
        region=args.region,
        profile=args.profile,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix or "",
    )

    created = save_tables(tables, out_path)

    # Minimal console output
    print(f"Extracted {len(tables)} table(s). Wrote {len(created)} file(s):")
    for p in created:
        print(f" - {p}")


if __name__ == "__main__":
    main()

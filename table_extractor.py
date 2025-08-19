#!/usr/bin/env python3
"""
AWS Textract Table Extractor

This script extracts tables from PDFs using AWS Textract's AnalyzeDocument API
with TABLES feature for optimal table extraction.
"""

import os
import sys
import json
import logging
import boto3
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('table_extractor.log')
    ]
)
logger = logging.getLogger(__name__)

# Default settings
DEFAULT_REGION = "us-east-2"
DEFAULT_OUTPUT_DIR = "output"

@dataclass
class TableCell:
    """Represents a single cell in a table"""
    row_index: int
    col_index: int
    text: str
    row_span: int = 1
    col_span: int = 1
    is_header: bool = False
    confidence: float = 100.0

class TableExtractor:
    """Handles table extraction from PDFs using AWS Textract"""
    
    def __init__(self, region_name: str = DEFAULT_REGION):
        """Initialize the Textract client"""
        self.textract = boto3.client('textract', region_name=region_name)
        self.blocks_cache = {}
    
    def process_pdf(self, pdf_path: str) -> List[pd.DataFrame]:
        """Process a PDF file and extract tables"""
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Convert PDF to images (one page at a time)
        images = self._pdf_to_images(pdf_path)
        if not images:
            logger.error("Failed to convert PDF to images")
            return []
        
        all_tables = []
        
        # Process each page
        for i, image_path in enumerate(images):
            logger.info(f"Processing page {i+1}/{len(images)}")
            
            # Analyze document with Textract
            with open(image_path, 'rb') as image_file:
                image_bytes = image_file.read()
            
            try:
                # First, try with TABLES feature
                response = self.textract.analyze_document(
                    Document={'Bytes': image_bytes},
                    FeatureTypes=['TABLES']
                )
                
                # Save the raw response for debugging
                self._save_response(response, f"response_page_{i+1}.json")
                
                # Extract tables from the response
                page_tables = self._extract_tables_from_response(response)
                
                if page_tables:
                    all_tables.extend(page_tables)
                    logger.info(f"Extracted {len(page_tables)} tables from page {i+1}")
                else:
                    logger.warning(f"No tables found in page {i+1}")
                
            except Exception as e:
                logger.error(f"Error processing page {i+1}: {str(e)}")
                continue
            
            # Clean up the image file
            try:
                os.remove(image_path)
            except:
                pass
        
        return all_tables
    
    def _pdf_to_images(self, pdf_path: str, dpi: int = 300) -> List[str]:
        """Convert PDF to images (one per page)"""
        try:
            from pdf2image import convert_from_path
            
            # Create output directory if it doesn't exist
            os.makedirs("temp_images", exist_ok=True)
            
            # Convert PDF to images
            images = convert_from_path(pdf_path, dpi=dpi)
            
            # Save images to temporary files
            image_paths = []
            for i, image in enumerate(images):
                image_path = f"temp_images/page_{i+1}.png"
                image.save(image_path, 'PNG')
                image_paths.append(image_path)
            
            return image_paths
            
        except ImportError:
            logger.error("pdf2image is not installed. Please install it with: pip install pdf2image")
            return []
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            return []
    
    def _save_response(self, response: Dict, filename: str):
        """Save the Textract response to a JSON file"""
        try:
            os.makedirs("debug", exist_ok=True)
            with open(f"debug/{filename}", 'w') as f:
                json.dump(response, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save response: {str(e)}")
    
    def _extract_tables_from_response(self, response: Dict) -> List[pd.DataFrame]:
        """Extract tables from Textract response"""
        # First, build a cache of all blocks by ID
        self.blocks_cache = {block['Id']: block for block in response['Blocks']}
        
        # Find all tables in the response
        tables = []
        for block in response['Blocks']:
            if block['BlockType'] == 'TABLE':
                table = self._extract_table(block)
                if table is not None:
                    tables.append(table)
        
        return tables
    
    def _extract_table(self, table_block: Dict) -> Optional[pd.DataFrame]:
        """Extract a single table from a TABLE block"""
        # Get all cells in this table
        cells = []
        relationships = table_block.get('Relationships', [])
        
        for relationship in relationships:
            if relationship['Type'] == 'CHILD':
                for cell_id in relationship['Ids']:
                    cell_block = self.blocks_cache.get(cell_id)
                    if cell_block and cell_block['BlockType'] == 'CELL':
                        cell = self._extract_cell(cell_block)
                        if cell:
                            cells.append(cell)
        
        if not cells:
            return None
        
        # Determine table dimensions
        max_row = max(cell.row_index for cell in cells)
        max_col = max(cell.col_index for cell in cells)
        
        # Create empty DataFrame with the right dimensions
        df = pd.DataFrame(
            index=range(1, max_row + 1),
            columns=range(1, max_col + 1)
        )
        
        # Fill in the DataFrame with cell values
        for cell in cells:
            for r in range(cell.row_index, cell.row_index + cell.row_span):
                for c in range(cell.col_index, cell.col_index + cell.col_span):
                    if r <= max_row and c <= max_col:
                        df.at[r, c] = cell.text if (r == cell.row_index and c == cell.col_index) else ""
        
        # Clean up the DataFrame
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Use first row as header if it looks like a header
        if not df.empty and len(df) > 1:
            first_row = df.iloc[0].astype(str).str.lower()
            if any('column' in str(x).lower() or 'header' in str(x).lower() for x in first_row):
                df.columns = df.iloc[0]
                df = df[1:].reset_index(drop=True)
        
        return df
    
    def _extract_cell(self, cell_block: Dict) -> Optional[TableCell]:
        """Extract cell information from a CELL block"""
        try:
            # Get cell position and span
            row_index = cell_block.get('RowIndex', 0)
            col_index = cell_block.get('ColumnIndex', 0)
            row_span = cell_block.get('RowSpan', 1)
            col_span = cell_block.get('ColumnSpan', 1)
            
            # Get cell content
            text = ""
            relationships = cell_block.get('Relationships', [])
            
            for relationship in relationships:
                if relationship['Type'] == 'CHILD':
                    for child_id in relationship['Ids']:
                        child_block = self.blocks_cache.get(child_block_id, {})
                        if child_block.get('BlockType') == 'WORD':
                            if text:
                                text += " "
                            text += child_block.get('Text', '')
            
            # If no text was found, try to get it from the cell block directly
            if not text and 'Text' in cell_block:
                text = cell_block['Text']
            
            return TableCell(
                row_index=row_index,
                col_index=col_index,
                text=text.strip(),
                row_span=row_span,
                col_span=col_span,
                is_header=cell_block.get('EntityTypes', []) == ['COLUMN_HEADER'],
                confidence=cell_block.get('Confidence', 0)
            )
            
        except Exception as e:
            logger.warning(f"Error extracting cell: {str(e)}")
            return None

def save_tables(tables: List[pd.DataFrame], output_dir: str = DEFAULT_OUTPUT_DIR):
    """Save extracted tables to CSV files"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, table in enumerate(tables):
        if table is not None and not table.empty:
            output_path = os.path.join(output_dir, f"table_{i+1}.csv")
            table.to_csv(output_path, index=False)
            logger.info(f"Saved table to {output_path}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract tables from PDF using AWS Textract')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, help='Output directory for extracted tables')
    parser.add_argument('--region', default=DEFAULT_REGION, help='AWS region')
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.pdf_path):
        logger.error(f"File not found: {args.pdf_path}")
        return 1
    
    try:
        # Initialize the extractor
        extractor = TableExtractor(region_name=args.region)
        
        # Extract tables from the PDF
        tables = extractor.process_pdf(args.pdf_path)
        
        if not tables:
            logger.warning("No tables were extracted from the PDF")
            return 1
        
        # Save the extracted tables
        save_tables(tables, args.output_dir)
        
        logger.info(f"Successfully extracted {len(tables)} tables from {args.pdf_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())

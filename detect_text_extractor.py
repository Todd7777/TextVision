#!/usr/bin/env python3
"""
Text Detection Extractor using AWS Textract

This script extracts text from PDFs using AWS Textract's DetectDocumentText API
and attempts to structure it into tables.
"""

import os
import sys
import json
import logging
import boto3
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('detect_text_extractor.log')
    ]
)
logger = logging.getLogger(__name__)

class TextDetectExtractor:
    """Extracts text from PDFs using AWS Textract's DetectDocumentText API"""
    
    def __init__(self, region_name: str = "us-east-2"):
        """Initialize the Textract client"""
        self.textract = boto3.client('textract', region_name=region_name)
    
    def extract_text(self, pdf_path: str) -> Dict:
        """Extract text from a PDF file"""
        if not os.path.isfile(pdf_path):
            logger.error(f"File not found: {pdf_path}")
            return {}
        
        logger.info(f"Extracting text from: {pdf_path}")
        
        try:
            # Read the PDF file as bytes
            with open(pdf_path, 'rb') as file:
                pdf_bytes = file.read()
            
            # Call Textract DetectDocumentText API
            response = self.textract.detect_document_text(
                Document={'Bytes': pdf_bytes}
            )
            
            # Save the raw response for debugging
            self._save_response(response, "textract_response.json")
            
            return response
            
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return {}
    
    def process_text_to_tables(self, response: Dict) -> List[pd.DataFrame]:
        """Process Textract response into tables"""
        if not response.get('Blocks'):
            return []
        
        # Group text by line
        lines = []
        for block in response['Blocks']:
            if block['BlockType'] == 'LINE':
                lines.append({
                    'text': block.get('Text', '').strip(),
                    'bounding_box': block.get('Geometry', {}).get('BoundingBox', {})
                })
        
        # Simple heuristic: group lines into tables based on vertical alignment
        tables = self._group_lines_into_tables(lines)
        return tables
    
    def _group_lines_into_tables(self, lines: List[Dict]) -> List[pd.DataFrame]:
        """Group lines into tables based on vertical alignment"""
        if not lines:
            return []
        
        # Sort lines by vertical position
        sorted_lines = sorted(
            [line for line in lines if line.get('bounding_box')],
            key=lambda x: x['bounding_box'].get('Top', 0)
        )
        
        tables = []
        current_table = []
        
        # Simple grouping: assume lines close to each other vertically are in the same table
        prev_bottom = None
        for line in sorted_lines:
            if not line['text'].strip():
                continue
                
            current_top = line['bounding_box'].get('Top', 0)
            
            # If there's a significant gap, start a new table
            if prev_bottom is not None and (current_top - prev_bottom) > 0.02:  # Threshold for new table
                if current_table:
                    table_df = self._create_table_from_lines(current_table)
                    if not table_df.empty:
                        tables.append(table_df)
                    current_table = []
            
            current_table.append(line)
            prev_bottom = current_top + line['bounding_box'].get('Height', 0)
        
        # Add the last table
        if current_table:
            table_df = self._create_table_from_lines(current_table)
            if not table_df.empty:
                tables.append(table_df)
        
        return tables
    
    def _create_table_from_lines(self, lines: List[Dict]) -> pd.DataFrame:
        """Create a DataFrame from a list of text lines"""
        if not lines:
            return pd.DataFrame()
        
        # Simple approach: split each line by whitespace and create a DataFrame
        data = []
        for line in lines:
            if not line['text'].strip():
                continue
            
            # Split by multiple spaces to handle aligned columns
            row = [cell for cell in line['text'].split('  ') if cell.strip()]
            if row:
                data.append(row)
        
        # Find the maximum number of columns
        max_cols = max(len(row) for row in data) if data else 0
        
        # Pad rows with empty strings to ensure consistent column count
        data = [row + [''] * (max_cols - len(row)) for row in data]
        
        # Create DataFrame
        if data:
            # Use first row as header if it looks like a header
            if any(header in data[0][0].lower() for header in ['county', 'amendment', 'yes', 'no']):
                df = pd.DataFrame(data[1:], columns=data[0])
            else:
                df = pd.DataFrame(data)
            
            # Clean up the DataFrame
            df = df.dropna(how='all').dropna(axis=1, how='all')
            return df
        
        return pd.DataFrame()
    
    def _save_response(self, response: Dict, filename: str):
        """Save the Textract response to a JSON file"""
        try:
            os.makedirs("debug", exist_ok=True)
            with open(f"debug/{filename}", 'w') as f:
                json.dump(response, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Could not save response: {str(e)}")

def save_tables(tables: List[pd.DataFrame], output_dir: str = "output"):
    """Save extracted tables to CSV files"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, table in enumerate(tables):
        if table is not None and not table.empty:
            # Clean up the table
            table = table.dropna(how='all').dropna(axis=1, how='all')
            
            # Save to CSV
            output_path = os.path.join(output_dir, f"table_{i+1}.csv")
            table.to_csv(output_path, index=False)
            logger.info(f"Saved table to {output_path}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract text from PDF using AWS Textract')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--output-dir', default="output", 
                       help='Output directory for extracted tables (default: output)')
    parser.add_argument('--region', default="us-east-2", 
                       help=f'AWS region (default: us-east-2)')
    
    args = parser.parse_args()
    
    # Initialize the extractor
    extractor = TextDetectExtractor(region_name=args.region)
    
    # Extract text from the PDF
    response = extractor.extract_text(args.pdf_path)
    
    if not response:
        logger.error("Failed to extract text from the PDF")
        return 1
    
    # Process text into tables
    tables = extractor.process_text_to_tables(response)
    
    if not tables:
        logger.warning("No tables were extracted from the PDF")
        return 1
    
    # Save the extracted tables
    save_tables(tables, args.output_dir)
    
    logger.info(f"Successfully extracted {len(tables)} tables to {args.output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

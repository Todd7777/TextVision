#!/usr/bin/env python3
"""
PDF to Image Extractor using AWS Textract

This script converts PDF pages to images and then extracts text using AWS Textract.
"""

import os
import sys
import json
import logging
import boto3
import pandas as pd
from pdf2image import convert_from_path
from PIL import Image
import io
import tempfile
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pdf_image_extractor.log')
    ]
)
logger = logging.getLogger(__name__)

class PDFImageExtractor:
    """Extracts text from PDFs by first converting to images"""
    
    def __init__(self, region_name: str = "us-east-2"):
        """Initialize the Textract client"""
        self.textract = boto3.client('textract', region_name=region_name)
    
    def extract_tables(self, pdf_path: str, dpi: int = 300) -> List[pd.DataFrame]:
        """Extract tables from a PDF by first converting to images"""
        if not os.path.isfile(pdf_path):
            logger.error(f"File not found: {pdf_path}")
            return []
        
        logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            # Convert PDF to images
            images = self._convert_pdf_to_images(pdf_path, dpi=dpi)
            if not images:
                logger.error("Failed to convert PDF to images")
                return []
            
            all_tables = []
            
            # Process each image
            for i, image in enumerate(images):
                logger.info(f"Processing page {i+1}/{len(images)}")
                
                # Convert image to bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Call Textract
                response = self.textract.detect_document_text(
                    Document={'Bytes': img_byte_arr}
                )
                
                # Save the raw response for debugging
                self._save_response(response, f"textract_response_page_{i+1}.json")
                
                # Process text into tables
                tables = self._process_text_to_tables(response)
                if tables:
                    all_tables.extend(tables)
            
            return all_tables
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return []
    
    def _convert_pdf_to_images(self, pdf_path: str, dpi: int = 300) -> List[Image.Image]:
        """Convert PDF to images"""
        try:
            logger.info(f"Converting PDF to images (DPI: {dpi})...")
            images = convert_from_path(pdf_path, dpi=dpi)
            logger.info(f"Converted {len(images)} pages to images")
            return images
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            return []
    
    def _process_text_to_tables(self, response: Dict) -> List[pd.DataFrame]:
        """Process Textract response into tables"""
        if not response.get('Blocks'):
            return []
        
        # Extract lines with their bounding boxes
        lines = []
        for block in response['Blocks']:
            if block['BlockType'] == 'LINE':
                bbox = block.get('Geometry', {}).get('BoundingBox', {})
                lines.append({
                    'text': block.get('Text', '').strip(),
                    'top': bbox.get('Top', 0),
                    'left': bbox.get('Left', 0),
                    'width': bbox.get('Width', 0),
                    'height': bbox.get('Height', 0)
                })
        
        # Sort lines by vertical position
        lines.sort(key=lambda x: (x['top'], x['left']))
        
        # Group lines into tables based on vertical alignment
        tables = self._group_lines_into_tables(lines)
        return tables
    
    def _group_lines_into_tables(self, lines: List[Dict]) -> List[pd.DataFrame]:
        """Group lines into tables based on vertical alignment"""
        if not lines:
            return []
        
        # Simple grouping by vertical position
        tables = []
        current_table = []
        
        for i, line in enumerate(lines):
            if not line['text']:
                continue
                
            # If current line is too far from the previous one, start a new table
            if current_table and (line['top'] - (current_table[-1]['top'] + current_table[-1]['height'])) > 0.05:
                table_df = self._create_table_from_lines(current_table)
                if not table_df.empty:
                    tables.append(table_df)
                current_table = []
                
            current_table.append(line)
        
        # Add the last table
        if current_table:
            table_df = self._create_table_from_lines(current_table)
            if not table_df.empty:
                tables.append(table_df)
        
        return tables
    
    def _create_table_from_lines(self, lines: List[Dict]) -> pd.DataFrame:
        """Create a DataFrame from a list of text lines with improved structure detection"""
        if not lines:
            return pd.DataFrame()
        
        # First, try to detect if this is a structured table
        is_structured = self._is_structured_table([line['text'] for line in lines])
        
        if is_structured:
            # For structured tables, use fixed-width splitting
            data = []
            for line in lines:
                if not line['text'].strip():
                    continue
                
                # Split by fixed positions (assuming columns are aligned)
                row = []
                text = line['text']
                
                # Try to split by consistent spacing
                parts = [p for p in text.split('  ') if p.strip()]
                if len(parts) >= 3:  # At least 3 columns
                    row = [p.strip() for p in parts]
                else:
                    # Fallback to simple space splitting
                    row = text.split()
                
                if row:
                    data.append(row)
        else:
            # For less structured content, use simpler splitting
            data = []
            for line in lines:
                if not line['text'].strip():
                    continue
                data.append([line['text'].strip()])
        
        if not data:
            return pd.DataFrame()
        
        # Find the maximum number of columns
        max_cols = max(len(row) for row in data) if data else 0
        
        # Pad rows with empty strings to ensure consistent column count
        data = [row + [''] * (max_cols - len(row)) for row in data]
        
        # Create DataFrame
        if data:
            # Use first row as header if it looks like a header
            if len(data) > 1 and self._is_header_row(data[0]):
                df = pd.DataFrame(data[1:], columns=data[0])
            else:
                df = pd.DataFrame(data)
            
            # Clean up the DataFrame
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            # Additional cleaning
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            
            # Remove empty rows and columns
            df = df.loc[df.astype(str).ne('').any(axis=1)]
            df = df.loc[:, df.astype(str).ne('').any(axis=0)]
            
            return df
        
        return pd.DataFrame()
    
    def _is_structured_table(self, lines: List[str]) -> bool:
        """Check if the text appears to be a structured table"""
        if len(lines) < 3:  # Need at least 3 lines to determine structure
            return False
        
        # Count the number of spaces in each line
        space_counts = [line.count('  ') for line in lines if line.strip()]
        
        # If most lines have multiple double spaces, it's likely a table
        if len(space_counts) > 2 and sum(c > 1 for c in space_counts) / len(space_counts) > 0.7:
            return True
            
        return False
    
    def _is_header_row(self, row: List[str]) -> bool:
        """Check if a row appears to be a header"""
        if not row:
            return False
            
        # Check for common header words
        header_indicators = ['county', 'amendment', 'yes', 'no', 'total', 'precinct', 'votes']
        return any(any(indicator in str(cell).lower() for indicator in header_indicators) 
                  for cell in row if isinstance(cell, str))
    
    def _save_response(self, response: Dict, filename: str):
        """Save the Textract response to a JSON file"""
        try:
            os.makedirs("debug", exist_ok=True)
            with open(f"debug/{filename}", 'w') as f:
                json.dump(response, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Could not save response: {str(e)}")

def save_tables(tables: List[pd.DataFrame], output_dir: str = "output"):
    """Save extracted tables to CSV and Excel files with improved formatting"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, table in enumerate(tables):
        if table is not None and not table.empty:
            # Clean up the table
            table = table.dropna(how='all').dropna(axis=1, how='all')
            
            # Clean up column names
            table.columns = [str(col).strip() for col in table.columns]
            
            # Try to identify the main data table (the one with county voting data)
            if i == 0 and len(table.columns) >= 6:  # First table with multiple columns
                # Clean up the table structure
                table = clean_voting_table(table)
                
                # Save to CSV and Excel
                base_filename = f"election_results_1948"
            else:
                base_filename = f"table_{i+1}"
            
            # Save to CSV
            csv_path = os.path.join(output_dir, f"{base_filename}.csv")
            table.to_csv(csv_path, index=False)
            
            # Save to Excel with formatting
            excel_path = os.path.join(output_dir, f"{base_filename}.xlsx")
            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                table.to_excel(writer, index=False, sheet_name='Election Results')
                
                # Get the workbook and worksheet objects
                workbook = writer.book
                worksheet = writer.sheets['Election Results']
                
                # Add a header format
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'fg_color': '#D7E4BC',
                    'border': 1
                })
                
                # Write the column headers with the defined format
                for col_num, value in enumerate(table.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                
                # Adjust column widths
                for i, col in enumerate(table.columns):
                    max_length = max(\
                        table[col].astype(str).apply(len).max(),
                        len(str(col))
                    ) + 2  # Add a little extra space
                    worksheet.set_column(i, i, min(max_length, 30))
            
            logger.info(f"Saved table to {csv_path} and {excel_path}")

def clean_voting_table(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and restructure the voting results table"""
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Check if the first row contains the amendment information
    if "VOTE ON PROPOSED CONSTITUTIONAL AMENDMENTS" in df.iloc[0, 0]:
        # Remove the header rows
        df = df.iloc[2:].reset_index(drop=True)
    
    # Set column names if they exist in the first row
    if any(x in str(df.iloc[0, 0]).lower() for x in ['county', 'amendment']):
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)
    
    # Clean up column names
    df.columns = [str(col).strip() for col in df.columns]
    
    # Remove any rows where all values are empty
    df = df.dropna(how='all')
    
    # Convert numeric columns to numeric, coerce errors to NaN
    for col in df.columns:
        if col.lower() != 'county':
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    
    # Calculate totals if they don't exist
    if 'Total' not in df.columns and len(df.columns) >= 3:
        numeric_cols = [col for col in df.columns if col != 'County']
        if numeric_cols:
            df['Total'] = df[numeric_cols].sum(axis=1)
    
    # Sort by county name if it exists
    if 'County' in df.columns:
        df = df.sort_values('County')
    
    return df

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract tables from PDF by converting to images first')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--output-dir', default="output", 
                       help='Output directory for extracted tables (default: output)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for image conversion (default: 300)')
    parser.add_argument('--region', default="us-east-2", 
                       help=f'AWS region (default: us-east-2)')
    
    args = parser.parse_args()
    
    # Initialize the extractor
    extractor = PDFImageExtractor(region_name=args.region)
    
    # Extract tables from the PDF
    tables = extractor.extract_tables(args.pdf_path, dpi=args.dpi)
    
    if not tables:
        logger.error("No tables were extracted from the PDF")
        return 1
    
    # Save the extracted tables
    save_tables(tables, args.output_dir)
    
    logger.info(f"Successfully extracted {len(tables)} tables to {args.output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

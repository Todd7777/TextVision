#!/usr/bin/env python3
"""
Async PDF Table Extractor using AWS Textract

This script extracts tables from PDFs using AWS Textract's StartDocumentAnalysis API,
which is the recommended approach for PDF documents.
"""

import os
import sys
import json
import logging
import boto3
import pandas as pd
import time
from botocore.exceptions import ClientError
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pdf_extractor.log')
    ]
)
logger = logging.getLogger(__name__)

class PDFTableExtractor:
    """Extracts tables from PDFs using AWS Textract's async API"""
    
    def __init__(self, region_name: str = "us-east-2", s3_bucket: str = None):
        """Initialize the Textract and S3 clients"""
        self.region_name = region_name
        self.s3_bucket = s3_bucket
        self.textract = boto3.client('textract', region_name=region_name)
        self.s3 = boto3.client('s3', region_name=region_name)
    
    def upload_to_s3(self, file_path: str, object_key: str = None) -> str:
        """Upload a file to S3 and return the object key"""
        if not object_key:
            object_key = os.path.basename(file_path)
            
        try:
            self.s3.upload_file(file_path, self.s3_bucket, object_key)
            logger.info(f"Uploaded {file_path} to s3://{self.s3_bucket}/{object_key}")
            return object_key
        except ClientError as e:
            logger.error(f"Error uploading to S3: {str(e)}")
            raise
    
    def start_document_analysis(self, s3_object_key: str) -> str:
        """Start asynchronous document analysis and return the job ID"""
        try:
            response = self.textract.start_document_analysis(
                DocumentLocation={
                    'S3Object': {
                        'Bucket': self.s3_bucket,
                        'Name': s3_object_key
                    }
                },
                FeatureTypes=['TABLES']
            )
            
            job_id = response.get('JobId')
            if not job_id:
                raise ValueError("No JobId in response")
                
            logger.info(f"Started document analysis job: {job_id}")
            return job_id
            
        except ClientError as e:
            logger.error(f"Error starting document analysis: {str(e)}")
            raise
    
    def wait_for_job_completion(self, job_id: str, poll_interval: int = 5) -> Dict:
        """Wait for the Textract job to complete and return the result"""
        status = 'IN_PROGRESS'
        result = None
        
        logger.info("Waiting for Textract job to complete...")
        
        while status == 'IN_PROGRESS':
            time.sleep(poll_interval)
            response = self.textract.get_document_analysis(JobId=job_id)
            status = response.get('JobStatus', 'FAILED')
            
            if status == 'SUCCEEDED':
                result = response
                break
            elif status in ['FAILED', 'PARTIAL_SUCCESS']:
                status_reason = response.get('StatusMessage', 'Unknown error')
                raise Exception(f"Textract job {status.lower()}: {status_reason}")
        
        logger.info("Textract job completed successfully")
        return result
    
    def delete_s3_object(self, object_key: str):
        """Delete the S3 object after processing"""
        try:
            self.s3.delete_object(Bucket=self.s3_bucket, Key=object_key)
            logger.info(f"Deleted s3://{self.s3_bucket}/{object_key}")
        except ClientError as e:
            logger.warning(f"Error deleting S3 object: {str(e)}")
    
    def extract_tables(self, pdf_path: str) -> List[pd.DataFrame]:
        """Extract tables from a PDF file using Textract's async API"""
        if not os.path.isfile(pdf_path):
            logger.error(f"File not found: {pdf_path}")
            return []
        
        if not self.s3_bucket:
            logger.error("S3 bucket name is required for PDF processing")
            return []
        
        logger.info(f"Extracting tables from: {pdf_path}")
        
        try:
            # Upload the PDF to S3
            object_key = self.upload_to_s3(pdf_path)
            
            try:
                # Start the document analysis job
                job_id = self.start_document_analysis(object_key)
                
                # Wait for the job to complete and get the results
                response = self.wait_for_job_completion(job_id)
                
                # Save the raw response for debugging
                self._save_response(response, "textract_response.json")
                
                # Extract tables from the response
                tables = self._extract_tables_from_response(response)
                
                if not tables:
                    logger.warning("No tables found in the document")
                else:
                    logger.info(f"Extracted {len(tables)} tables from the document")
                
                return tables
                
            finally:
                # Clean up the S3 object
                self.delete_s3_object(object_key)
                
        except Exception as e:
            logger.error(f"Error extracting tables: {str(e)}")
            return []
    
    def _save_response(self, response: Dict, filename: str):
        """Save the Textract response to a JSON file"""
        try:
            os.makedirs("debug", exist_ok=True)
            with open(f"debug/{filename}", 'w') as f:
                json.dump(response, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Could not save response: {str(e)}")
    
    def _extract_tables_from_response(self, response: Dict) -> List[pd.DataFrame]:
        """Extract tables from Textract response"""
        # Build a cache of all blocks by ID for quick lookup
        self.blocks_cache = {block['Id']: block for block in response.get('Blocks', [])}
        
        tables = []
        
        # Find all TABLE blocks in the response
        for block in response.get('Blocks', []):
            if block['BlockType'] == 'TABLE':
                table = self._extract_table(block)
                if table is not None and not table.empty:
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
                    cell_block = self.blocks_cache.get(cell_id, {})
                    if cell_block.get('BlockType') == 'CELL':
                        cell = self._extract_cell(cell_block)
                        if cell:
                            cells.append(cell)
        
        if not cells:
            return None
        
        # Determine table dimensions
        max_row = max(cell['row_index'] for cell in cells)
        max_col = max(cell['col_index'] for cell in cells)
        
        # Create empty DataFrame with the right dimensions
        df = pd.DataFrame(
            index=range(1, max_row + 1),
            columns=range(1, max_col + 1)
        )
        
        # Fill in the DataFrame with cell values
        for cell in cells:
            row_span = cell.get('row_span', 1)
            col_span = cell.get('col_span', 1)
            
            for r in range(cell['row_index'], cell['row_index'] + row_span):
                for c in range(cell['col_index'], cell['col_index'] + col_span):
                    if r <= max_row and c <= max_col:
                        # Only set the text in the first cell of a merged cell
                        df.at[r, c] = cell['text'] if (r == cell['row_index'] and c == cell['col_index']) else ""
        
        # Clean up the DataFrame
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Reset index to ensure proper row numbering
        df = df.reset_index(drop=True)
        
        return df
    
    def _extract_cell(self, cell_block: Dict) -> Optional[Dict]:
        """Extract cell information from a CELL block"""
        try:
            # Get cell position and span
            row_index = cell_block.get('RowIndex', 0)
            col_index = cell_block.get('ColumnIndex', 0)
            row_span = cell_block.get('RowSpan', 1)
            col_span = cell_block.get('ColumnSpan', 1)
            
            # Initialize text content
            text_parts = []
            
            # Get cell content from relationships
            for relationship in cell_block.get('Relationships', []):
                if relationship['Type'] == 'CHILD':
                    for child_id in relationship['Ids']:
                        child_block = self.blocks_cache.get(child_id, {})
                        if child_block.get('BlockType') in ['WORD', 'SELECTION_ELEMENT', 'LINE']:
                            text_parts.append(child_block.get('Text', ''))
            
            # Combine text parts
            text = ' '.join(text_parts).strip()
            
            # If no text was found, try to get it from the cell block directly
            if not text and 'Text' in cell_block:
                text = cell_block['Text']
            
            return {
                'row_index': row_index,
                'col_index': col_index,
                'row_span': row_span,
                'col_span': col_span,
                'text': text,
                'is_header': 'COLUMN_HEADER' in cell_block.get('EntityTypes', [])
            }
            
        except Exception as e:
            logger.warning(f"Error extracting cell: {str(e)}")
            return None

def save_tables(tables: List[pd.DataFrame], output_dir: str = "output"):
    """Save extracted tables to CSV files"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, table in enumerate(tables):
        if table is not None and not table.empty:
            # Clean up the table
            table = table.dropna(how='all').dropna(axis=1, how='all')
            
            # Use first row as header if it looks like a header
            if not table.empty and len(table) > 1:
                first_row = table.iloc[0].astype(str).str.lower()
                if any(header in str(x).lower() for x in first_row for header in ['column', 'header', 'county', 'amendment']):
                    table.columns = table.iloc[0]
                    table = table[1:].reset_index(drop=True)
            
            # Save to CSV
            output_path = os.path.join(output_dir, f"table_{i+1}.csv")
            table.to_csv(output_path, index=False)
            logger.info(f"Saved table to {output_path}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract tables from PDF using AWS Textract')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--s3-bucket', required=True, help='S3 bucket name for temporary storage')
    parser.add_argument('--output-dir', default="output", 
                       help='Output directory for extracted tables (default: output)')
    parser.add_argument('--region', default="us-east-2", 
                       help=f'AWS region (default: us-east-2)')
    
    args = parser.parse_args()
    
    # Initialize the extractor
    extractor = PDFTableExtractor(region_name=args.region, s3_bucket=args.s3_bucket)
    
    # Extract tables from the PDF
    tables = extractor.extract_tables(args.pdf_path)
    
    if not tables:
        logger.error("No tables were extracted from the PDF")
        return 1
    
    # Save the extracted tables
    save_tables(tables, args.output_dir)
    
    logger.info(f"Successfully extracted {len(tables)} tables to {args.output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

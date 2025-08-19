import os
import tempfile
import streamlit as st
import boto3
import pandas as pd
from pdf2image import convert_from_path
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="PDF Table Extractor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'tables' not in st.session_state:
    st.session_state.tables = []
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# Initialize AWS client
textract = boto3.client('textract', region_name='us-east-2')

def extract_table_structure(response):
    """Extract table structure from Textract response"""
    # Get the text blocks
    blocks = response['Blocks']
    
    # Get the relationships between blocks
    block_map = {block['Id']: block for block in blocks}
    
    tables = []
    
    # Find all TABLE blocks
    for block in blocks:
        if block['BlockType'] == 'TABLE':
            table = []
            
            # Get all CELL blocks that are children of this TABLE
            for relationship in block.get('Relationships', []):
                if relationship['Type'] == 'CHILD':
                    for cell_id in relationship['Ids']:
                        cell = block_map[cell_id]
                        if cell['BlockType'] == 'CELL':
                            # Get the text in the cell
                            cell_text = ''
                            # Check if cell has content
                            if 'Relationships' in cell:
                                for rel in cell['Relationships']:
                                    if rel['Type'] == 'CHILD':
                                        for id in rel['Ids']:
                                            word = block_map[id]
                                            if word['BlockType'] == 'WORD':
                                                cell_text += word['Text'] + ' '
                            
                            # Add cell to the table
                            row = cell.get('RowIndex', 1) - 1  # Convert to 0-based index
                            col = cell.get('ColumnIndex', 1) - 1  # Convert to 0-based index
                            
                            # Ensure table has enough rows
                            while len(table) <= row:
                                table.append([])
                            
                            # Ensure row has enough columns
                            while len(table[row]) <= col:
                                table[row].append('')
                            
                            table[row][col] = cell_text.strip()
            
            if table:
                tables.append(pd.DataFrame(table[1:], columns=table[0]))
    
    return tables

def process_pdf_with_textract(pdf_path):
    """Process PDF using AWS Textract"""
    try:
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=400)
        all_tables = []
        
        for i, image in enumerate(images):
            # Save image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG', quality=100)
            img_byte_arr = img_byte_arr.getvalue()
            
            # Call Textract with table detection
            response = textract.analyze_document(
                Document={'Bytes': img_byte_arr},
                FeatureTypes=['TABLES', 'FORMS']
            )
            
            # Extract tables with structure
            page_tables = extract_table_structure(response)
            all_tables.extend(page_tables)
            
            # If no tables found, fall back to line-based extraction
            if not page_tables:
                response = textract.detect_document_text(
                    Document={'Bytes': img_byte_arr}
                )
                lines = []
                for item in response.get('Blocks', []):
                    if item['BlockType'] == 'LINE':
                        lines.append(item['Text'])
                
                if lines:
                    all_tables.append(pd.DataFrame(lines, columns=['Text']))
        
        return all_tables
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return []

def main():
    st.title("📄 PDF Table Extractor")
    st.write("Upload a PDF file to extract tables using AWS Textract")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        
        # Display file info
        st.sidebar.subheader("File Info")
        st.sidebar.write(f"Name: {uploaded_file.name}")
        st.sidebar.write(f"Size: {uploaded_file.size / 1024:.2f} KB")
        
        # Process button
        if st.button("Extract Tables"):
            with st.spinner("Processing PDF..."):
                # Save uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    # Process the PDF
                    st.session_state.tables = process_pdf_with_textract(tmp_path)
                finally:
                    # Clean up the temporary file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
    
    # Display results
    if st.session_state.tables:
        st.subheader("Extracted Tables")
        
        for i, table in enumerate(st.session_state.tables):
            # Make column names unique by appending a number to duplicates
            table.columns = [f"{col}_{i}" if list(table.columns).count(col) > 1 else col 
                           for i, col in enumerate(table.columns)]
            
            with st.expander(f"Table {i+1} ({len(table)} rows)"):
                # Display the table with unique column names
                st.dataframe(table, use_container_width=True)
                
                # For download, use the original column names without the suffixes
                download_df = table.copy()
                download_df.columns = [col.split('_')[0] for col in download_df.columns]
                
                # Download button for each table
                csv = download_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"Download Table {i+1} as CSV",
                    data=csv,
                    file_name=f"table_{i+1}.csv",
                    mime="text/csv",
                    key=f"download_{i}"
                )
        
        # Download all tables as Excel
        if len(st.session_state.tables) > 1:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                for i, table in enumerate(st.session_state.tables):
                    # For Excel, use original column names without suffixes
                    excel_df = table.copy()
                    excel_df.columns = [col.split('_')[0] for col in excel_df.columns]
                    excel_df.to_excel(writer, sheet_name=f'Table_{i+1}', index=False)
            
            st.download_button(
                label="Download All Tables as Excel",
                data=excel_buffer.getvalue(),
                file_name="all_tables.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main()

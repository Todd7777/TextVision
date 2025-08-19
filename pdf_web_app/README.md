# PDF Table Extractor Web App

A Flask web application that extracts tables from PDF files using AWS Textract.

## Prerequisites

- Python 3.7+
- AWS Account with Textract access
- AWS credentials configured (either via `~/.aws/credentials` or environment variables)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd pdf_web_app
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your AWS credentials:
   ```
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_REGION=your_region
   FLASK_SECRET_KEY=your_secret_key_here
   ```

## Running the Application

1. Start the Flask development server:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

## Usage

1. Click "Upload PDF" and select a PDF file containing tables
2. Wait for the processing to complete
3. Download the extracted tables in Excel format

## Features

- Simple, intuitive web interface
- Supports multiple tables per PDF
- Downloads results in Excel format
- Responsive design works on desktop and mobile

## File Structure

```
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── static/                # Static files (CSS, JS, images)
│   └── styles.css         # Custom styles
├── templates/             # HTML templates
│   ├── base.html          # Base template
│   ├── index.html         # Upload form
│   └── results.html       # Results page
├── uploads/               # Temporary storage for uploaded files
└── output/                # Extracted tables
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

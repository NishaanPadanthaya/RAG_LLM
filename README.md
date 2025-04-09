# PDF Question Answering App

This Streamlit application allows users to upload PDF documents and ask questions about their content. The app extracts text and tables from the PDF, creates a vector database, and uses Groq LLM to answer questions based on the document content.

## Features

- Upload PDF documents
- Extract text and tables from PDFs
- Ask questions about the document content
- Get AI-generated answers using Groq LLM

## Requirements

- Python 3.8+
- Groq API key (sign up at https://console.groq.com/)
- Java Runtime Environment (JRE) for tabula-py

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run streamlit_app.py
   ```

2. Open your web browser and navigate to the URL displayed in the terminal (usually http://localhost:8501)

3. Enter your Groq API key in the sidebar

4. Upload a PDF document

5. Ask questions about the document content

## How It Works

1. **PDF Processing**:
   - Text extraction using PyPDF2
   - Table extraction using tabula-py
   - Text chunking for better processing

2. **Vector Database Creation**:
   - Text embedding using sentence-transformers
   - Vector storage using FAISS for efficient similarity search

3. **Question Answering**:
   - Retrieval of relevant document chunks based on the question
   - Answer generation using Groq LLM

## Notes

- The quality of answers depends on the quality of the PDF and the extracted text
- For PDFs with complex layouts or scanned images, the extraction might not be perfect
- The app works best with text-based PDFs with clear formatting



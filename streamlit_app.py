import os
import tempfile
import streamlit as st
import PyPDF2
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage

# Load environment variables from .env file
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="PDF Question Answering App",
    page_icon="📄",
    layout="wide"
)

# App title and description
st.title("📄 PDF Question Answering App")
st.markdown("""
Upload a PDF document and ask questions about its content.
The app will extract text from the PDF, create a vector database,
and use Groq LLM to answer your questions based on the document content.
""")

# Get API key from environment variable
api_key = os.getenv("GROQ_API_KEY", "")
if not api_key:
    st.error("GROQ_API_KEY not found in .env file. Please add your API key to the .env file.")

# Sidebar for model selection
with st.sidebar:
    st.header("Configuration")
    model_name = st.selectbox(
        "Select Groq Model:",
        ["llama-3.2-1b-preview", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        index=0
    )
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app uses:
    - PyPDF2 for text extraction
    - FAISS for vector search
    - Groq LLM for question answering
    """)

# Functions for PDF processing
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyPDF2."""
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def preprocess_text(text):
    """Splits text into chunks for indexing."""
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return text_splitter.split_text(text)

def process_pdf(pdf_path):
    """Process PDF and create vector database."""
    with st.spinner("Extracting text from PDF..."):
        raw_text = extract_text_from_pdf(pdf_path)

        # Preprocess text
        text_chunks = preprocess_text(raw_text)

        # Create embeddings
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Create FAISS vector store
        vector_db = FAISS.from_texts(text_chunks, embedding_model)

        return vector_db, len(text_chunks)

def retrieve_and_generate(vector_db, query, api_key, model_name):
    """Retrieves relevant document chunks and generates a response using Groq LLM."""
    # Initialize LLM
    llm = ChatGroq(model_name=model_name, groq_api_key=api_key)

    # Search in FAISS vector database
    search_results = vector_db.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in search_results])

    # Prepare messages for LLM
    messages = [
        SystemMessage(content="You are a helpful AI assistant that answers document queries accurately based on the provided context. If the answer cannot be found in the context, say so."),
        HumanMessage(content=f"Using the following document context, answer the query:\n\n{context}\n\nQuery: {query}")
    ]

    # Generate response using Groq LLM
    response = llm(messages)
    return response.content

# Main app logic
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        pdf_path = tmp_file.name

    # Process the PDF
    if 'vector_db' not in st.session_state:
        try:
            vector_db, num_text_chunks = process_pdf(pdf_path)
            st.session_state.vector_db = vector_db
            st.session_state.pdf_processed = True

            # Display success message
            st.success(f"PDF processed successfully! Extracted {num_text_chunks} text chunks.")
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            st.session_state.pdf_processed = False

    # Question answering section
    if st.session_state.get('pdf_processed', False):
        st.markdown("## Ask Questions")
        query = st.text_input("Enter your question about the document:")

        if query:
            if st.button("Get Answer"):
                if not api_key:
                    st.error("GROQ_API_KEY not found in .env file. Please add your API key to the .env file.")
                else:
                    with st.spinner("Generating answer..."):
                        try:
                            answer = retrieve_and_generate(st.session_state.vector_db, query, api_key, model_name)

                            # Display the answer
                            st.markdown("### Answer")
                            st.markdown(answer)
                        except Exception as e:
                            st.error(f"Error generating answer: {str(e)}")

    # Clean up the temporary file
    os.unlink(pdf_path)
else:
    st.info("Please upload a PDF document to get started.")

# Add a footer
st.markdown("---")
st.markdown("Built with Streamlit, LangChain, and Groq LLM")

import pdfplumber
import re
import os
import logging
from pathlib import Path
from nltk.tokenize import sent_tokenize
import nltk
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("pdf_extract.log")]
)
logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download("punkt_tab", quiet=True)

# Configuration
PDF_PATHS = [
    Path(r"C:\Users\nrksi\OneDrive\Desktop\CHATBOT TELECOM UPDATED\BWA guidelines.pdf"),
    Path(r"C:\Users\nrksi\OneDrive\Desktop\CHATBOT TELECOM UPDATED\Guidelines for MNPS.pdf"),
    Path(r"C:\Users\nrksi\OneDrive\Desktop\CHATBOT TELECOM UPDATED\IGW Licensing Guidelines.pdf"),
    Path(r"C:\Users\nrksi\OneDrive\Desktop\CHATBOT TELECOM UPDATED\Instructions for Issuance of Registration Certificate for the Operation of BPO_Call Center.pdf"),
    Path(r"C:\Users\nrksi\OneDrive\Desktop\CHATBOT TELECOM UPDATED\IPTSP Guideline.pdf"),
    Path(r"C:\Users\nrksi\OneDrive\Desktop\CHATBOT TELECOM UPDATED\NTTN _L_L.pdf"),
    Path(r"C:\Users\nrksi\OneDrive\Desktop\CHATBOT TELECOM UPDATED\Regulatory and Licensing Guideline for Internet Service Provider (ISP) in Bangladesh.pdf"),
    Path(r"C:\Users\nrksi\OneDrive\Desktop\CHATBOT TELECOM UPDATED\Regulatory and Licensing Guidelines for Satellite Operator in Bangladesh.pdf"),
    Path(r"C:\Users\nrksi\OneDrive\Desktop\CHATBOT TELECOM UPDATED\Submarine Cable_licensing_guideline.pdf"),
    Path(r"C:\Users\nrksi\OneDrive\Desktop\CHATBOT TELECOM UPDATED\vehicle_tracking_services_guidelines.pdf"),
    Path(r"C:\Users\nrksi\OneDrive\Desktop\CHATBOT TELECOM UPDATED\VSAT Guideline.pdf")
]

# API keys from environment variables
OPENAI_API_KEY = "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"  # Replace with your OpenAI API key
QDRANT_API_KEY = "Zqa0jFliFIUN0GP42Pp7Jy6P-AxpRjM18-oBYFUQdsn5MmWWBlKnEg"
QDRANT_CLUSTER_URL = "https://6c8299ea-71fe-4a63-a916-6e961520b10a.us-east-1-0.aws.cloud.qdrant.io"
COLLECTION_NAME = "Chatbot_Telecom1"
CHUNK_SIZE = 1024
OVERLAP = int(CHUNK_SIZE * 0.2)  # 20% overlap
VECTOR_SIZE = 1536  # For text-embedding-ada-002

# Initialize clients
try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    exit(1)

try:
    qdrant_client = QdrantClient(url=QDRANT_CLUSTER_URL, api_key=QDRANT_API_KEY)
except Exception as e:
    logger.error(f"Failed to initialize Qdrant client: {e}")
    exit(1)

# Extract text from a single PDF, preserving financial terms and numbers
def extract_pdf_text(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                lines = page_text.split("\n")
                for line in lines:
                    line = line.strip()
                    if line:
                        text += line + "\n"
                        # Expanded regex for financial terms
                        if re.search(r'\$\d+\.?\d*|\b\d+\.?\d*\b|\bBDT\b|\bfee\b|\bcost\b|\bcharge\b|\bprice\b|\btariff\b|\bpayment\b|\brenewal\b|\blicense\b|\bregistration\b|\bsubscription\b|\bdeposit\b|\bpenalty\b|\bsurcharge\b', line, re.IGNORECASE):
                            text += line + "\n"
            return text
    except Exception as e:
        logger.error(f"Error extracting {pdf_path}: {e}")
        return ""

# Chunk text with sentence boundaries and overlap
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    current_length = 0
    overlap_text = ""

    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length <= chunk_size:
            current_chunk += sentence + " "
            current_length += sentence_length + 1
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                overlap_start = max(0, len(current_chunk) - overlap)
                overlap_text = current_chunk[overlap_start:]
                current_chunk = overlap_text + sentence + " "
                current_length = len(current_chunk)
            else:
                current_chunk = sentence + " "
                current_length = sentence_length + 1

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Generate embeddings with OpenAI text-embedding-ada-002
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(Exception)
)
def generate_embeddings(chunks):
    try:
        batch_size = 48  # Adjust based on OpenAI rate limits
        embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            response = openai_client.embeddings.create(
                input=batch,
                model="text-embedding-ada-002"
            )
            embeddings.extend([embedding.embedding for embedding in response.data])
            time.sleep(1)  # Avoid rate limits
        return embeddings, VECTOR_SIZE
    except Exception as e:
        logger.error(f"OpenAI embedding failed: {e}")
        return [], VECTOR_SIZE

# Store embeddings in Qdrant
def store_in_qdrant(embeddings, chunks, pdf_name, pdf_index, vector_size):
    try:
        # Check if collection exists and verify configuration
        if not qdrant_client.collection_exists(COLLECTION_NAME):
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
        else:
            collection_info = qdrant_client.get_collection(COLLECTION_NAME)
            if collection_info.config.params.vectors.size != vector_size:
                logger.error(f"Collection {COLLECTION_NAME} has vector size {collection_info.config.params.vectors.size}, expected {vector_size}")
                return

        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={"text": chunk, "source": pdf_name, "pdf_index": pdf_index, "chunk_index": chunk_idx}
            )
            for chunk_idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
        logger.info(f"Stored {len(chunks)} chunks from {pdf_name}")
    except Exception as e:
        logger.error(f"Error storing in Qdrant for {pdf_name}: {e}")

# Process all PDFs
def process_pdfs():
    try:
        errors = []
        for pdf_index, path in enumerate(PDF_PATHS):
            name = path.name
            logger.info(f"Processing {name}")
            if not os.path.exists(path):
                logger.error(f"Not found: {path}")
                errors.append(f"Not found: {path}")
                continue
            text = extract_pdf_text(path)
            if not text:
                errors.append(f"No text: {name}")
                continue
            chunks = chunk_text(text)
            embeddings, size = generate_embeddings(chunks)
            if not embeddings:
                errors.append(f"No embeddings: {name}")
                continue
            store_in_qdrant(embeddings, chunks, name, pdf_index, size)
        if errors:
            logger.warning(f"Errors: {errors}")
        else:
            logger.info("All PDFs processed")
    except Exception as e:
        logger.error(f"Processing failed: {e}")

if __name__ == "__main__":
    # Delete existing collection to ensure fresh data
    try:
        qdrant_client.delete_collection(COLLECTION_NAME)
        logger.info(f"Deleted collection {COLLECTION_NAME}")
    except Exception as e:
        logger.warning(f"Could not delete collection: {e}")
    process_pdfs()
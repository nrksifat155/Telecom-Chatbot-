from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI
from sentence_transformers import CrossEncoder
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import os
from dotenv import load_dotenv
import re
import os
import pandas as pd
# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("chatbot.log")]
)
logger = logging.getLogger(__name__)

# API keys and configuration
QDRANT_API_KEY = "Zqa0jFliFIUN0GP42Pp7Jy6P-AxpRjM18-oBYFUQdsn5MmWWBlKnEg"
OPENAI_API_KEY = "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"
QDRANT_CLUSTER_URL = "https://6c8299ea-71fe-4a63-a916-6e961520b10a.us-east-1-0.aws.cloud.qdrant.io"
COLLECTION_NAME = "Chatbot_Telecom1"
DEFAULT_QUESTION = "show all fees and charges"
VECTOR_SIZE = 1536  # Matches text-embedding-ada-002
EMBEDDING_MODEL = "text-embedding-ada-002"
RERANK_MODEL = "BAAI/bge-reranker-base"

# Initialize clients
try:
    qdrant_client = QdrantClient(url=QDRANT_CLUSTER_URL, api_key=QDRANT_API_KEY)
except Exception as e:
    logger.error(f"Failed to initialize Qdrant client: {e}")
    exit(1)

try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    exit(1)

try:
    rerank_model = CrossEncoder(RERANK_MODEL)
    logger.info("BGE reranker model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize CrossEncoder: {e}")
    rerank_model = None

# Perform semantic search in Qdrant
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=120),
    retry=retry_if_exception_type(Exception)
)
def search_qdrant(query, limit=200):
    try:
        response = openai_client.embeddings.create(input=[query], model=EMBEDDING_MODEL)
        query_embedding = response.data[0].embedding
        if len(query_embedding) != VECTOR_SIZE:
            raise ValueError(f"Query embedding size {len(query_embedding)} does not match expected {VECTOR_SIZE}")
        search_result = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=limit,
            with_payload=True,
        ).points
        if not search_result:
            logger.warning("No results found in Qdrant search")
            return []
        return [(hit.payload["text"], hit.payload["source"], hit.score) for hit in search_result]
    except Exception as e:
        logger.error(f"Error searching Qdrant: {e}")
        return []

# Rerank results using CrossEncoder with financial term filtering
def rerank_results(query, chunks):
    try:
        # Filter for financial terms
        filtered_chunks = [
            chunk for chunk in chunks
            if re.search(r'\$\d+\.?\d*|\b\d+\.?\d*\b|\bBDT\b|\bfee\b|\bcost\b|\bcharge\b|\bprice\b|\btariff\b|\bpayment\b|\brenewal\b|\blicense\b|\bregistration\b|\bsubscription\b|\bdeposit\b|\bpenalty\b|\bsurcharge\b|\blevy\b|\bassessment\b|\bfine\b', chunk[0], re.IGNORECASE)
        ]
        if not filtered_chunks:
            logger.warning("No chunks with financial terms found, using top 50 chunks")
            filtered_chunks = chunks[:50]  # Fallback to top 50 Qdrant-scored chunks
        documents = [chunk[0] for chunk in filtered_chunks]
        if not rerank_model:
            logger.warning("CrossEncoder not available, using Qdrant scores")
            return [{"text": chunk[0], "source": chunk[1], "relevance_score": chunk[2]} for chunk in filtered_chunks]
        scores = rerank_model.predict([(query, doc) for doc in documents])
        ranked = sorted(
            zip(filtered_chunks, scores),
            key=lambda x: x[1],
            reverse=True
        )[:30]  # Top 30 for broader fee coverage
        reranked = [{"text": chunk[0], "source": chunk[1], "relevance_score": float(score)} for (chunk, score) in ranked]
        logger.info(f"Reranked {len(reranked)} results with CrossEncoder")
        return reranked
    except Exception as e:
        logger.error(f"Error reranking with CrossEncoder: {e}")
        return [{"text": chunk[0], "source": chunk[1], "relevance_score": chunk[2]} for chunk in chunks[:50]]

# Generate answer using OpenAI
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=120),
    retry=retry_if_exception_type(Exception)
)
def generate_answer(query, reranked_results):
    try:
        context = "\n".join(
            f"Source: {result['source']}\nText: {result['text'][:1000]}..."
            for result in reranked_results[:15]  # Top 15 for broader context
        )
        prompt = (
         f"Context: {context}\n\n"
f"Question: {query}\n\n"
f"Analyze the provided context and answer the question strictly based on the specific category asked.\n"
f"- Provide the answer as a clear *bulleted list*, limited to **6–10 lines**.\n"
f"- Include **only information** that matches the question category exactly. Do not mix other types of data.\n"
f"    • If the question is about **Eligibility**, include only eligibility criteria such as who can apply, legal requirements, and disqualifications.\n"
f"      ❌ Do NOT include application steps, required documents, general obligations, or fees.\n"
f"    • If the question is about **Fees/Costs**, include only financial obligations: fee types, amounts, payment rules, penalties, etc.\n"
f"      ❌ Do NOT include eligibility, requirements, or procedures.\n"
f"    • If the question is about **General Requirements**, include only general legal, policy, and procedural obligations explicitly listed under that heading.\n"
f"      ❌ Do NOT mix in eligibility, technical specs, or fees unless directly part of that section.\n"
f"    • For **any other topic** (technical requirements, application process, license term, etc.), include only content that directly answers the question.\n"
f"- Each bullet must end with a citation in the format: `Source: dummy.pdf`\n"
f"- Do NOT explain, summarize, or add extra information outside the context.\n\n"
f"If no relevant information is found matching the question type, respond with:\n"
f"'No relevant information found in the provided context.'"

        )
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer based only on the provided context, extracting all financial obligations comprehensively, including vague or partial mentions, in a clear and structured bulleted list."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating answer with OpenAI: {e}")
        return "Unable to generate answer."



def log_to_excel(question, answer):
    file_path = "Test_Result.xlsx"
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
    else:
        df = pd.DataFrame(columns=["Question", "Answer"])
    
    new_row = pd.DataFrame([{"Question": question, "Answer": answer}])
    df = pd.concat([df, new_row], ignore_index=True)

    df.to_excel(file_path, index=False)



# Main search function
def search_and_rerank(query):
    initial_results = search_qdrant(query)
    if not initial_results:
        return {"answer": "No results found.", "results": []}
    reranked_results = rerank_results(query, initial_results)
    answer = generate_answer(query, reranked_results)
    log_to_excel(query, answer)
    return {"answer": answer}

# CLI interface with continuous interaction
def main():
    print("Telecom Chatbot: Type 'exit' to quit.")
    while True:
        query = input("Enter your question: ")
        if query.lower() == "exit":
            break
        logger.info(f"Processing question: {query}")
        result = search_and_rerank(query)
        print("\nAnswer:")
        print(result["answer"])
        print()

if __name__ == "__main__":
    main()
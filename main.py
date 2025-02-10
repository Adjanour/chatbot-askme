from fastapi import FastAPI, HTTPException,Request,Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, AsyncGenerator
from functools import lru_cache
import os
from openai import AsyncOpenAI
import uuid
from fastapi.responses import StreamingResponse
import json
import boto3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(title="RAG API")

# Allow only the specific domain in CORS (modify as needed)
ALLOWED_ORIGIN = "https://api.oziza.org"
# ALLOWED_ORIGIN = "*"
app.add_middleware(CORSMiddleware, allow_origins=[ALLOWED_ORIGIN], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])

def get_secret(secret_name):
    client = boto3.client('ssm',region_name='us-east-1')
    response = client.get_parameter(
        Name=secret_name,
        WithDecryption=True
    )
    return response['Parameter']['Value']

OPENAI_API_KEY = get_secret('OPENAI_API_KEY')
model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="faq_collection")
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

async def verify_origin(request: Request):
    origin = request.headers.get("Origin")  # Get request origin
    if origin != ALLOWED_ORIGIN:
        logger.warning(f"Unauthorized access attempt from {origin}")
        raise HTTPException(status_code=403, detail="Access denied")


class Query(BaseModel):
    text: str


@lru_cache(maxsize=100)
def get_embedding(text: str) -> List[float]:
    try:
        return model.encode(text, convert_to_numpy=True).tolist()
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        raise HTTPException(status_code=500, detail="Embedding generation failed")


async def load_faqs(file_path: str):
    try:
        with open(file_path, 'r') as f:
            faqs = json.load(f)

        questions = [faq['question'] for faq in faqs]
        embeddings = model.encode(questions, convert_to_numpy=True).tolist()

        collection.add(
            documents=questions,
            embeddings=embeddings,
            metadatas=[{"answer": faq['answer']} for faq in faqs],
            ids=[str(uuid.uuid4()) for _ in faqs]
        )
        logger.info(f"Loaded {len(faqs)} FAQ items")
    except Exception as e:
        logger.error(f"Error loading FAQs: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    try:
        if not os.path.exists("faqs.json"):
            logger.warning("FAQs file not found")
            return

        if os.path.exists("chroma_db"):
            # Check if the database is already populated
            existing_data_count = collection.count()
            if existing_data_count > 0:
                logger.info(f"Chroma DB already populated with {existing_data_count} items, skipping reload.")
                return

        # Load FAQs since the database is empty or doesn't exist
        await load_faqs("faqs.json")

    except Exception as e:
        logger.error(f"Startup error: {e}")


async def generate_answer(query: str, context: List[Dict]) -> str:
    try:
        prompt = f"""You are a knowledgeable and helpful AI assistant specializing in health-related queries. Provide clear, accurate, and concise responses based on reliable information. Ensure your answers are well-structured and informative, without referencing any sources or prior knowledge explicitly.
                FAQ Entries:
                {context}

                User Question: {query}

                Answer:"""

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise HTTPException(status_code=500, detail="Answer generation failed")

@app.get("/")
async def root():
    return {"chatbot-status":"healthy"}

@app.post("/query")
async def query_endpoint(query: Query):
    try:
        query_embedding = get_embedding(query.text)
        results = collection.query(query_embeddings=[query_embedding], n_results=3)
        context = [{"question": doc, "answer": meta["answer"]}
                   for doc, meta in zip(results['documents'][0], results['metadatas'][0])]
        answer = await generate_answer(query.text, context)
        return {"answer": answer, "sources": context}
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def stream_generate_answer(query: str, context: List[Dict]) -> AsyncGenerator[str, None]:
    try:
        prompt = f"""You are a knowledgeable and helpful AI assistant specializing in health-related queries. Provide clear, accurate, and concise responses based on reliable information. Ensure your answers are well-structured and informative, without referencing any sources or prior knowledge explicitly.
                FAQ Entries:
                {context}

                User Question: {query}

                Answer:"""

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
            stream=True
        )

        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@app.post("/stream", dependencies=[Depends(verify_origin)],)
async def stream_endpoint(query: Query):
    try:
        query_embedding = get_embedding(query.text)
        results = collection.query(query_embeddings=[query_embedding], n_results=3)
        context = [{"question": doc, "answer": meta["answer"]}
                   for doc, meta in zip(results['documents'][0], results['metadatas'][0])]

        return StreamingResponse(
            stream_generate_answer(query.text, context),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8100)

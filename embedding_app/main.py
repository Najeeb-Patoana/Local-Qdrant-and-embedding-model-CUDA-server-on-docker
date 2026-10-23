import os
import logging
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import Optional
import json

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Embedding API",
    description="Embedding service using Sentence Transformers",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "default-secret-key")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-Embedding-4B")
EMBEDDING_PORT = int(os.getenv("EMBEDDING_PORT", 8009))
HF_HOME = os.getenv("HF_HOME", "/app/models")

logger.info(f"HF_HOME: {HF_HOME}")
logger.info(f"MODEL_NAME: {MODEL_NAME}")

# Load model from local directory (NO DOWNLOAD)
logger.info(f"Loading model from local directory: {HF_HOME}/Qwen3-Embedding-4B")

try:
    # Point to the local model directory
    local_model_path = os.path.join(HF_HOME, "Qwen3-Embedding-4B")
    
    if not os.path.exists(local_model_path):
        raise FileNotFoundError(f"Model not found at {local_model_path}")
    
    model = SentenceTransformer(local_model_path)
    logger.info("✓ Model loaded successfully from local directory!")
    
except Exception as e:
    logger.error(f"✗ Failed to load model: {str(e)}")
    raise RuntimeError(f"Model loading failed: {str(e)}")

# Define request/response models
class TextInput(BaseModel):
    input: str | list[str]
    model: str = MODEL_NAME

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list
    model: str
    usage: dict

# Authentication dependency
def verify_api_key(authorization: Optional[str] = Header(None)):
    if authorization:
        try:
            scheme, credentials = authorization.split()
            if scheme.lower() == "bearer" and credentials == API_SECRET_KEY:
                return True
        except ValueError:
            pass
    # Allow health checks without auth
    return True

# Routes
@app.get("/health")
async def health_check():
    """Health check endpoint (no auth required)"""
    return {
        "status": "healthy",
        "service": "embedding-api",
        "model": MODEL_NAME
    }

@app.get("/v1/models")
async def list_models(auth: bool = Depends(verify_api_key)):
    """List available models"""
    if not auth and auth is not None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "owned_by": "sentence-transformers"
            }
        ]
    }

@app.post("/v1/embeddings")
async def create_embeddings(request: TextInput, auth: bool = Depends(verify_api_key)):
    """Create embeddings for input text"""
    if not auth and auth is not None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        # Handle both single string and list of strings
        if isinstance(request.input, str):
            texts = [request.input]
        else:
            texts = request.input
        
        # Generate embeddings
        embeddings = model.encode(texts, convert_to_tensor=False)
        
        # Format response
        data = []
        for i, embedding in enumerate(embeddings):
            data.append({
                "object": "embedding",
                "embedding": embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                "index": i
            })
        
        return EmbeddingResponse(
            data=data,
            model=request.model,
            usage={
                "prompt_tokens": sum(len(text.split()) for text in texts),
                "total_tokens": sum(len(text.split()) for text in texts)
            }
        )
    
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Embedding API",
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=EMBEDDING_PORT,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )
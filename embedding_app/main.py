import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import List, Union
from pathlib import Path

from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch

# ✅ IMPORT SENTENCE TRANSFORMERS AT TOP LEVEL (NOT IN LIFESPAN)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logger_temp = logging.getLogger(__name__)
    logger_temp.error(f"Failed to import SentenceTransformer: {e}")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Ensure UTF-8 encoding on Windows
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

# ==================== LOGGING SETUP ====================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# ==================== GLOBAL STATE ====================

model = None
device = None

# ==================== LIFECYCLE MANAGEMENT ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global model, device
    
    try:
        # STARTUP
        logger.info("=" * 60)
        logger.info("🚀 Starting Embedding API Service")
        logger.info("=" * 60)
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError("SentenceTransformer not available")
        
        # Detect device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"✓ Device: {device}")
        
        if device == "cuda":
            try:
                logger.info(f"  CUDA Version: {torch.version.cuda}")
                logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
            except Exception as e:
                logger.warning(f"Could not get GPU details: {e}")
        
        # Load model
        model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-Embedding-4B")
        logger.info(f"\n📦 Loading model: {model_name}")
        logger.info(f"   This may take 2-5 minutes on first run...")
        
        try:
            model = SentenceTransformer(
                model_name,
                device=device,
                cache_folder=str(Path.home() / ".cache" / "huggingface")
            )
            
            # Test embedding
            test_embedding = model.encode("test", convert_to_tensor=False)
            logger.info(f"✓ Model loaded successfully!")
            logger.info(f"  Dimensions: {len(test_embedding)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}", exc_info=True)
            raise
        
        logger.info("=" * 60)
        logger.info("✅ API Ready for Requests")
        logger.info("=" * 60 + "\n")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}", exc_info=True)
        raise
    
    finally:
        # SHUTDOWN
        logger.info("\n" + "=" * 60)
        logger.info("🛑 Shutting down Embedding API")
        logger.info("=" * 60)
        
        try:
            if model is not None:
                del model
                logger.info("✓ Model unloaded")
            
            if device == "cuda":
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    logger.info("✓ CUDA memory cleared")
                except Exception as e:
                    logger.warning(f"Error clearing CUDA memory: {e}")
            
            logger.info("✅ Shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}", exc_info=True)

# ==================== FASTAPI APPLICATION ====================

app = FastAPI(
    title="Production Embedding API",
    description="High-performance embedding service with Qdrant integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== SECURITY ====================

security = HTTPBearer()
API_SECRET_KEY = os.getenv("API_SECRET_KEY")

if not API_SECRET_KEY:
    logger.warning("⚠️  API_SECRET_KEY not set - using default (development only!)")
    API_SECRET_KEY = "dev-key-change-in-production"

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """Verify Bearer token"""
    if credentials.credentials != API_SECRET_KEY:
        logger.warning(f"⚠️  Invalid token attempt")
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    return credentials.credentials

# ==================== MODELS ====================

class EmbeddingRequest(BaseModel):
    """Embedding request"""
    input: Union[str, List[str]] = Field(..., description="Text or texts to embed")
    model: str = Field(default_factory=lambda: os.getenv("MODEL_NAME", "Qwen/Qwen3-Embedding-4B"))
    
    class Config:
        json_schema_extra = {
            "example": {
                "input": "Hello world",
                "model": "Qwen/Qwen3-Embedding-4B"
            }
        }

class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: dict

# ==================== ENDPOINTS ====================

@app.get("/", tags=["Status"])
async def root() -> dict:
    """Root endpoint"""
    return {
        "name": "Production Embedding API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "model": os.getenv("MODEL_NAME", "Qwen/Qwen3-Embedding-4B")
    }

@app.get("/health", tags=["Status"])
async def health_check() -> dict:
    """Health check endpoint (no auth required)"""
    if model is None:
        logger.warning("Health check: Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        return {
            "status": "healthy",
            "model": os.getenv("MODEL_NAME", "Qwen/Qwen3-Embedding-4B"),
            "device": str(device),
            "timestamp": str(__import__('datetime').datetime.utcnow())
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/v1/models", tags=["Models"])
async def list_models(token: str = Depends(verify_token)) -> dict:
    """List available models"""
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-Embedding-4B")
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "owned_by": "Qwen",
                "created": 1699564160,
                "permission": [],
                "root": model_name,
                "parent": None
            }
        ]
    }

@app.post("/v1/embeddings", response_model=EmbeddingResponse, tags=["Embeddings"])
async def create_embedding(
    body: EmbeddingRequest,
    token: str = Depends(verify_token)
) -> dict:
    """Create embeddings"""
    
    if model is None:
        logger.error("Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Normalize input
        texts = [body.input] if isinstance(body.input, str) else body.input
        
        # Validate input
        if not texts:
            raise HTTPException(status_code=400, detail="Empty input")
        
        if len(texts) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 texts per request")
        
        logger.info(f"Processing {len(texts)} embeddings")
        
        # Generate embeddings
        with torch.no_grad():
            embeddings = model.encode(
                texts,
                convert_to_tensor=False,
                batch_size=32,
                show_progress_bar=False
            )
        
        # Format response
        data = [
            EmbeddingData(object="embedding", index=i, embedding=emb.tolist())
            for i, emb in enumerate(embeddings)
        ]
        
        logger.info(f"✓ Generated {len(data)} embeddings")
        
        return EmbeddingResponse(
            object="list",
            data=data,
            model=body.model,
            usage={
                "prompt_tokens": sum(len(t.split()) for t in texts),
                "total_tokens": sum(len(t.split()) for t in texts)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Embedding error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", tags=["Status"])
async def get_stats(token: str = Depends(verify_token)) -> dict:
    """Get API statistics"""
    try:
        gpu_memory = None
        if device == "cuda":
            try:
                gpu_memory = {
                    "allocated": float(torch.cuda.memory_allocated() / 1e9),
                    "reserved": float(torch.cuda.memory_reserved() / 1e9),
                    "total": float(torch.cuda.get_device_properties(0).total_memory / 1e9)
                }
            except:
                pass
        
        return {
            "device": str(device),
            "gpu_memory_gb": gpu_memory,
            "model": os.getenv("MODEL_NAME"),
            "timestamp": str(__import__('datetime').datetime.utcnow())
        }
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info"
    )
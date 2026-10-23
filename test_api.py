import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get ports from .env
NGINX_PORT = os.getenv("NGINX_PORT", "85")
EMBEDDING_PORT = os.getenv("EMBEDDING_PORT", "8009")
QDRANT_REST_PORT = os.getenv("QDRANT_REST_PORT", "6331")

# Base URLs
NGINX_BASE = f"http://localhost:{NGINX_PORT}"
EMBEDDING_API = f"http://localhost:{EMBEDDING_PORT}"
QDRANT_API = f"http://localhost:{QDRANT_REST_PORT}"
PUBLIC_BASE = "https://germinate-marvelous-outclass.ngrok-free.dev"

# Auth keys
API_KEY = os.getenv("API_SECRET_KEY", "N_prod_embed_secretkey")
QDRANT_KEY = os.getenv("QDRANT_API_KEY", "N_prod_qdrant_secretkey")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

qdrant_headers = {
    "api-key": QDRANT_KEY
}

def test_endpoints(base_url, name):
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    # Health
    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        print(f"Health: {r.status_code}")
    except Exception as e:
        print(f"Health: {e}")
    
    # List Models
    try:
        r = requests.get(f"{base_url}/v1/models", headers=headers, timeout=10)
        print(f"Models: {r.status_code}")
    except Exception as e:
        print(f"Models: {e}")
    
    # Single Embedding
    try:
        body = {"input": "Hello world", "model": "Qwen/Qwen3-Embedding-4B"}
        r = requests.post(f"{base_url}/v1/embeddings", json=body, headers=headers, timeout=30)
        data = r.json()
        dims = len(data['data'][0]['embedding']) if r.status_code == 200 else 0
        print(f"Embedding: {r.status_code} - Dimensions: {dims}")
    except Exception as e:
        print(f"Embedding: {e}")
    
    # Batch Embedding
    try:
        body = {"input": ["Hello", "World", "Test"], "model": "Qwen/Qwen3-Embedding-4B"}
        r = requests.post(f"{base_url}/v1/embeddings", json=body, headers=headers, timeout=30)
        count = len(r.json().get('data', [])) if r.status_code == 200 else 0
        print(f"Batch: {r.status_code} - Count: {count}")
    except Exception as e:
        print(f"Batch: {e}")

def test_qdrant(base_url, name):
    print(f"\n{'='*60}")
    print(f"Qdrant Testing: {name}")
    print(f"{'='*60}")
    
    try:
        r = requests.get(f"{base_url}/health")
        print(f"Health: {r.status_code}")
    except Exception as e:
        print(f"Health: {e}")
    
    try:
        r = requests.get(f"{base_url}/api/collections", headers=qdrant_headers)
        print(f" Collections: {r.status_code}")
    except Exception as e:
        print(f" Collections: {e}")

# Run all tests
print("\n╔════════════════════════════════════════════════════════════╗")
print("║           AI STACK - COMPREHENSIVE TEST SUITE             ║")
print("╚════════════════════════════════════════════════════════════╝")

print(f"\nPort Configuration:")
print(f"  Nginx: {NGINX_PORT}")
print(f"  Embedding API: {EMBEDDING_PORT}")
print(f"  Qdrant REST: {QDRANT_REST_PORT}")

# Test via Nginx (recommended)
test_endpoints(NGINX_BASE, f"NGINX (localhost:{NGINX_PORT})")

# Test direct Embedding API
test_endpoints(EMBEDDING_API, f"Direct Embedding API (localhost:{EMBEDDING_PORT})")

# Test Qdrant
test_qdrant(QDRANT_API, f"Qdrant (localhost:{QDRANT_REST_PORT})")

# Test public (if ngrok is running)
test_endpoints(PUBLIC_BASE, "PUBLIC (Ngrok)")

print("\n All tests complete!")
print("\n╔════════════════════════════════════════════════════════════╗")
print("║              ACCESS URLS WITH ENV PORTS                   ║")
print("╚════════════════════════════════════════════════════════════╝")
print(f"Health:       http://localhost:{NGINX_PORT}/health")
print(f"Docs:         http://localhost:{NGINX_PORT}/docs")
print(f"Embeddings:   http://localhost:{NGINX_PORT}/v1/embeddings")
print(f"Qdrant:       http://localhost:{NGINX_PORT}/")
print(f"Direct Emb:   http://localhost:{EMBEDDING_PORT}/health")
print(f"Direct Qdr:   http://localhost:{QDRANT_REST_PORT}/health")
print(f"Public:       {PUBLIC_BASE}/health")
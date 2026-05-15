# Academic AI Chatbot Backend

Sistem backend chatbot cerdas untuk lingkungan akademik menggunakan teknik **Retrieval-Augmented Generation (RAG)**.

## 🛠️ Tech Stack

| Komponen | Teknologi |
|----------|-----------|
| Framework | FastAPI (Python 3.10+) |
| Orchestration | LangChain |
| Vector Database | ChromaDB (Local Persistence) |
| LLM Model | Llama 3.2 3B (via Ollama) |
| Embeddings | HuggingFace BGE-M3 |

## 📁 Struktur Proyek

```
app/
├── api/v1/          # Endpoint definitions
│   ├── chat.py      # Chat endpoint (+ streaming SSE)
│   ├── ingest.py    # Document upload & ingestion
│   └── health.py    # Health check
├── core/
│   └── config.py    # Application settings
├── services/
│   ├── ai_logic.py  # LangChain RAG pipeline
│   └── vector_store.py  # Document processing & ChromaDB
├── schemas/
│   └── models.py    # Pydantic request/response models
├── repositories/
│   └── chroma_repo.py   # ChromaDB data access layer
└── main.py          # Application entry point
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running
- NVIDIA GPU with 6GB+ VRAM (recommended) or CPU

### 2. Install Ollama & Model

```bash
# Install Ollama (see https://ollama.com/download)
# Pull the Llama 3.2 3B model
ollama pull llama3.2:3b
```

### 3. Setup Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env as needed (defaults should work for local development)
```

### 5. Run the Server

```bash
python -m app.main
```

Server will start at `http://localhost:8000`

## 📡 API Endpoints

### Health Check
```
GET /api/v1/health
GET /api/v1/health/collection
```

### Document Ingestion
```
POST /api/v1/ingest          # Upload document (PDF/MD/TXT)
GET  /api/v1/ingest/info     # Collection statistics
DELETE /api/v1/ingest/{filename}  # Delete document
```

### Chat
```
POST /api/v1/chat            # Full JSON response
POST /api/v1/chat/stream     # Streaming SSE response
```

### Example Chat Request

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Bagaimana cara mengajukan cuti akademik?"}'
```

### Example Ingest Request

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@panduan_akademik.pdf" \
  -F "category=akademik"
```

## 📖 API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `LLM_MODEL_NAME` | `llama3.2:3b` | LLM model name |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | HuggingFace embedding model |
| `CHROMA_PERSIST_DIR` | `./chroma_data` | ChromaDB storage path |
| `CHUNK_SIZE` | `1000` | Text chunk size |
| `CHUNK_OVERLAP` | `200` | Chunk overlap |
| `TOP_K_RESULTS` | `5` | Number of retrieved docs |

## 🔒 Privasi

Seluruh pemrosesan data (LLM & Vector DB) dilakukan secara **lokal** pada infrastruktur server internal untuk menjamin kerahasiaan data akademik.

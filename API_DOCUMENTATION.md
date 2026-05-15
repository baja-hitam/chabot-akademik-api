# API Documentation - Dual LLM Provider Support

## Overview

Semua endpoint RAG sudah terintegrasi dengan **dual LLM provider support**. API endpoint tetap sama, provider dipilih via `.env` configuration.

---

## Chat Endpoint

### POST `/api/v1/chat/query`

Kirim pertanyaan akademik dan dapatkan jawaban berdasarkan dokumen yang tersimpan.

**Request:**
```json
{
  "question": "Bagaimana cara mendaftar mata kuliah?",
  "category": "akademik"  // optional
}
```

**Response (200 OK):**
```json
{
  "answer": "Berdasarkan dokumentasi akademik kami, cara mendaftar mata kuliah adalah...",
  "sources": [
    {
      "content": "Lorem ipsum...",
      "source": "Panduan Akademik 2024",
      "category": "akademik",
      "relevance_score": 0.95,
      "document_year": 2024,
      "is_latest": true,
      "ocr_used": false
    }
  ],
  "processing_time": 1.234
}
```

**Provider Impact:**
- **Local (Ollama)**: Processing time 2-5 detik (CPU) atau 0.5-1.5 detik (GPU)
- **Groq Cloud**: Processing time 0.3-0.8 detik (konsisten & cepat)

---

## Chat Streaming Endpoint

### GET `/api/v1/chat/stream`

Stream jawaban token-by-token menggunakan Server-Sent Events (SSE).

**Query Parameters:**
```
?question=Bagaimana+cara+mendaftar+mata+kuliah?&category=akademik
```

**Response (200 OK - text/event-stream):**
```
data: Berdasarkan

data: dokumentasi

data: akademik

data: kami,

...
```

**Provider Impact:**
- **Local (Ollama)**: Stream lambat di CPU, cepat di GPU
- **Groq Cloud**: Stream sangat cepat, real-time response

---

## Health Check Endpoint

### GET `/api/v1/health`

Cek status aplikasi dan koneksi LLM.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "timestamp": "2024-05-13T10:30:00Z",
  "dependencies": {
    "llm": "healthy",
    "vector_store": "healthy",
    "embedding_model": "healthy"
  },
  "provider": "groq"  // atau "local"
}
```

---

## Ingest Endpoint

### POST `/api/v1/ingest/documents`

Upload dan ingest dokumen akademik (PDF format).

**Request:**
```multipart/form-data
file: <PDF file>
category: akademik
year: 2024
is_latest: true
```

**Response (200 OK):**
```json
{
  "status": "success",
  "document_id": "doc_123456",
  "chunks_created": 45,
  "category": "akademik",
  "processing_time": 5.234
}
```

**Note:** Ingest process TIDAK dipengaruhi oleh LLM provider (gunakan embedding model untuk vectorization).

---

## Code Example

### Python (Async)

```python
import httpx

async def query_chatbot():
    async with httpx.AsyncClient() as client:
        # Query dengan provider yang dikonfigurasi (auto-detect)
        response = await client.post(
            "http://localhost:8000/api/v1/chat/query",
            json={
                "question": "Apa persyaratan akademik?",
                "category": "akademik"
            }
        )
        
        result = response.json()
        print(f"Answer: {result['answer']}")
        print(f"Time: {result['processing_time']}s")
        print(f"Sources: {len(result['sources'])} documents")
        
        # Informasi provider tidak di-expose, tapi bisa di-check
        # via health endpoint
        health = await client.get(
            "http://localhost:8000/api/v1/health"
        )
        provider = health.json()['provider']
        print(f"Active Provider: {provider}")

# Run
import asyncio
asyncio.run(query_chatbot())
```

### Python (Streaming)

```python
import httpx

async def stream_answer():
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "GET",
            "http://localhost:8000/api/v1/chat/stream",
            params={
                "question": "Jelaskan proses akademik",
                "category": "akademik"
            }
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    token = line[6:]
                    print(token, end="", flush=True)

import asyncio
asyncio.run(stream_answer())
```

### JavaScript (Fetch API)

```javascript
// Standard request
async function queryChat() {
  const response = await fetch('http://localhost:8000/api/v1/chat/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question: 'Bagaimana cara mendaftar?',
      category: 'akademik'
    })
  });
  
  const data = await response.json();
  console.log(data.answer);
  console.log(`Response time: ${data.processing_time}s`);
}

// Streaming with EventSource
function streamChat() {
  const eventSource = new EventSource(
    'http://localhost:8000/api/v1/chat/stream?question=Jelaskan%20akademik'
  );
  
  eventSource.onmessage = (event) => {
    process.stdout.write(event.data);
  };
  
  eventSource.onerror = () => {
    eventSource.close();
  };
}
```

### cURL

```bash
# Standard query
curl -X POST http://localhost:8000/api/v1/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Apa saja mata kuliah wajib?",
    "category": "akademik"
  }'

# Streaming
curl -N http://localhost:8000/api/v1/chat/stream \
  -G --data-urlencode "question=Jelaskan proses akademik" \
  --data-urlencode "category=akademik"

# Health check
curl http://localhost:8000/api/v1/health
```

---

## Performance Comparison

### Example Query: "Bagaimana cara mendaftar mata kuliah?"

#### Local Ollama (CPU)
```json
{
  "answer": "[Full response...]",
  "processing_time": 3.456,
  "sources": 5
}
```

#### Groq Cloud API
```json
{
  "answer": "[Full response...]",
  "processing_time": 0.567,
  "sources": 5
}
```

**Groq is ~6x faster** dalam contoh ini!

---

## Error Handling

### Jika LLM tidak available

**Local (Ollama) - Connection Refused:**
```json
{
  "detail": "Failed to connect to Ollama at http://localhost:11434"
}
```
**Solution:** Pastikan `ollama serve` berjalan

**Groq - Invalid API Key:**
```json
{
  "detail": "Groq API key invalid or expired"
}
```
**Solution:** Update GROQ_API_KEY di .env

### Jika tidak ada dokumen

```json
{
  "answer": "Maaf, saya tidak menemukan informasi tersebut dalam dokumen yang tersedia...",
  "sources": [],
  "processing_time": 0.123
}
```

---

## Rate Limiting

### Local Ollama
- Unlimited (tergantung resource)
- Typically: 1-3 req/s (GPU), 0.2-0.5 req/s (CPU)

### Groq Cloud
- Free tier: 1000 requests/day
- Rate limit header: `x-ratelimit-remaining`
- Error 429: Quota exceeded

```bash
# Check rate limit
curl -i http://localhost:8000/api/v1/health | grep x-ratelimit
```

---

## Configuration Impact on API

| Setting | Local Ollama | Groq Cloud |
|---------|---|---|
| Response Time | Slower | Faster |
| Concurrent Requests | Limited | Better |
| Cost | Free | Free tier limit |
| Setup | Requires Ollama | Requires API key |
| Offline | ✓ Yes | ✗ No |

---

## Monitoring

```bash
# Check active provider
curl http://localhost:8000/api/v1/health | grep provider

# Monitor performance
watch -n 1 'curl -s http://localhost:8000/api/v1/health | jq .'

# Test with different categories
curl -X POST http://localhost:8000/api/v1/chat/query \
  -H "Content-Type: application/json" \
  -d '{"question": "test", "category": "akademik"}'
```

---

## Next Steps

1. ✅ Pilih provider di `.env`
2. ✅ Test endpoint dengan cURL atau Postman
3. ✅ Integrate dengan frontend
4. ✅ Monitor performance
5. ✅ Optimize berdasarkan kebutuhan

Happy API usage! 🚀

# LLM Provider Setup Guide

Aplikasi ini mendukung **dua opsi LLM**:

1. **Local (Ollama)** - Menjalankan model lokal tanpa koneksi internet
2. **Groq Cloud API** - Menggunakan API cloud dengan model high-performance

---

## 📋 Perbandingan Kedua Opsi

| Aspek | Local (Ollama) | Groq Cloud API |
|-------|---|---|
| **Setup** | Install Ollama locally | Daftar akun di groq.com |
| **Biaya** | Gratis | Gratis tier (dengan limit) |
| **Internet** | Tidak perlu | Diperlukan |
| **Kecepatan** | Tergantung GPU lokal | Cepat (high-performance) |
| **Privasi Data** | Lokal sepenuhnya | Dikirim ke Groq servers |
| **Models** | Llama, Mistral, Phi, dll | Mixtral, Llama 2/3, Gemma |
| **Maintenance** | Self-hosted | Groq mengelola |

---

## 🔧 OPSI 1: Local Ollama (Rekomendasi untuk Development)

### Prerequisites
- RAM minimal 8GB (16GB+ disarankan)
- GPU optional tapi sangat disarankan untuk performa lebih baik
- Internet hanya untuk download model awal

### Langkah Setup

#### 1. Install Ollama
Download dari: https://ollama.ai

#### 2. Start Ollama Server
```bash
ollama serve
```
Server akan berjalan di `http://localhost:11434`

#### 3. Pull Model
Di terminal baru:
```bash
# Option A: Llama 3.2 (3B) - Lightweight, cocok untuk CPU
ollama pull llama3.2:3b

# Option B: Mistral - Lebih powerful
ollama pull mistral

# Option C: Neural Chat - Optimized untuk conversational AI
ollama pull neural-chat

# Option D: Llama 2 - Popular
ollama pull llama2
```

#### 4. Konfigurasi `.env`
```env
LLM_PROVIDER=local
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL_NAME=llama3.2:3b
```

#### 5. Jalankan Aplikasi
```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

### Troubleshooting Local

| Error | Solusi |
|-------|--------|
| `Connection refused: 11434` | Pastikan `ollama serve` sedang berjalan di terminal terpisah |
| `Model not found` | Jalankan `ollama pull llama3.2:3b` |
| `Out of memory` | Kurangi `num_predict` di config atau gunakan model lebih kecil (3b vs 7b) |
| `Very slow responses` | Menggunakan CPU. Coba dengan GPU atau model lebih kecil |

---

## 🌥️ OPSI 2: Groq Cloud API (Rekomendasi untuk Production)

### Kelebihan
- ✅ Response sangat cepat (inference on Groq TPU)
- ✅ Tidak perlu hardware mahal
- ✅ Scalable untuk production
- ✅ API tier gratis (1000 req/day)

### Langkah Setup

#### 1. Daftar Akun Groq
1. Buka https://console.groq.com
2. Sign up dengan email atau Google
3. Verify email

#### 2. Buat API Key
1. Pergi ke Settings → API Keys
2. Klik "Create New API Key"
3. Copy key dan simpan di tempat aman

#### 3. Konfigurasi `.env`
```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxx
GROQ_MODEL_NAME=mixtral-8x7b-32768
```

#### 4. Model yang Tersedia di Groq
```
mixtral-8x7b-32768    # Fastest (default)
llama2-70b-4096       # Powerful
gemma-7b-it           # General purpose
llama-3-70b-versatile # Versatile
```

#### 5. Jalankan Aplikasi
```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

### Troubleshooting Groq

| Error | Solusi |
|-------|--------|
| `Invalid API key` | Double-check GROQ_API_KEY di `.env` |
| `Rate limit exceeded` | Groq free tier terbatas 1000 req/day. Upgrade plan atau tunggu reset |
| `Authentication failed` | Pastikan GROQ_API_KEY tidak ada trailing whitespace |

---

## 🔄 Switching Between Providers

### Dari Local ke Groq
```env
# Dari:
LLM_PROVIDER=local

# Ke:
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_xxxxx
```

### Dari Groq ke Local
```env
# Dari:
LLM_PROVIDER=groq

# Ke:
LLM_PROVIDER=local
OLLAMA_BASE_URL=http://localhost:11434
```

Tidak perlu restart, aplikasi akan switch otomatis!

---

## 📊 Performance Comparison

### Response Time (Contoh dengan query akademik)

```
Local (Ollama):
  - Llama 3.2 (3B, CPU): 2-5 detik per response
  - Llama 3.2 (3B, GPU): 0.5-1.5 detik per response
  - Mistral (7B, GPU): 1-3 detik per response

Cloud (Groq):
  - Mixtral 8x7B: 0.3-0.8 detik per response
  - Llama 2 70B: 0.5-1 detik per response
```

---

## 💡 Recommendations

### Pilih Local (Ollama) jika:
- Development/testing environment
- Perlu offline capability
- Privacy concerns
- Budget terbatas untuk CPU+RAM

### Pilih Groq jika:
- Production environment
- Need consistent high performance
- Banyak concurrent users
- Fast response time critical
- Tidak perlu maintain hardware

---

## 📝 Monitoring & Debugging

### Check which provider is active
```python
from app.core.config import get_settings
settings = get_settings()
print(f"Using LLM Provider: {settings.LLM_PROVIDER}")
print(f"Model: {settings.OLLAMA_MODEL_NAME or settings.GROQ_MODEL_NAME}")
```

### Check LLM Health
```bash
curl http://localhost:8000/api/v1/health
```

Response akan menunjukkan status LLM connection.

---

## 🚀 Next Steps

1. Pilih provider yang sesuai
2. Follow setup steps di atas
3. Update `.env` file
4. Jalankan aplikasi
5. Test dengan chat endpoint

Happy chatting! 🎉

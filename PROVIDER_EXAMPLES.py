"""
Quick examples showing how to switch between LLM providers.
"""

# ═══════════════════════════════════════════════════════════════════════════════════
# Example 1: Using Local Ollama
# ═══════════════════════════════════════════════════════════════════════════════════

# .env configuration:
"""
LLM_PROVIDER=local
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL_NAME=llama3.2:3b
"""

# Then run:
"""
ollama serve  # In another terminal
uvicorn app.main:app --reload
"""


# ═══════════════════════════════════════════════════════════════════════════════════
# Example 2: Using Groq Cloud API
# ═══════════════════════════════════════════════════════════════════════════════════

# .env configuration:
"""
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxx
GROQ_MODEL_NAME=mixtral-8x7b-32768
"""

# Then run:
"""
uvicorn app.main:app --reload
"""


# ═══════════════════════════════════════════════════════════════════════════════════
# Example 3: Programmatically check which provider is configured
# ═══════════════════════════════════════════════════════════════════════════════════

from app.core.config import get_settings
from app.services.ai_logic import ai_logic_service

settings = get_settings()
provider = settings.LLM_PROVIDER

if provider == "groq":
    print(f"Using Groq: {settings.GROQ_MODEL_NAME}")
    print(f"API Key configured: {bool(settings.GROQ_API_KEY)}")
else:
    print(f"Using Local Ollama: {settings.OLLAMA_MODEL_NAME}")
    print(f"Base URL: {settings.OLLAMA_BASE_URL}")

# Get LLM instance
llm = ai_logic_service._get_llm()
print(f"LLM type: {type(llm).__name__}")  # ChatGroq or ChatOllama


# ═══════════════════════════════════════════════════════════════════════════════════
# Example 4: Using the RAG service
# ═══════════════════════════════════════════════════════════════════════════════════

import asyncio

async def example_query():
    question = "Bagaimana cara mendaftar mata kuliah?"
    
    # Method 1: Get full response with sources
    response = await ai_logic_service.get_answer(question)
    print(f"Answer: {response.answer}")
    print(f"Sources: {len(response.sources)} documents")
    print(f"Time: {response.processing_time}s")
    
    # Method 2: Stream response (real-time)
    print("\nStreaming response:")
    async for chunk in ai_logic_service.stream_answer(question):
        print(chunk, end="", flush=True)

asyncio.run(example_query())


# ═══════════════════════════════════════════════════════════════════════════════════
# Example 5: Test both providers in sequence
# ═══════════════════════════════════════════════════════════════════════════════════

import asyncio
import os
from dotenv import load_dotenv

async def test_provider(provider_name):
    """Test a specific provider."""
    print(f"\nTesting {provider_name}...")
    
    # Set environment variable
    os.environ["LLM_PROVIDER"] = "local" if provider_name == "Ollama" else "groq"
    
    # Reset the cached settings
    from app.core.config import Settings
    Settings.model_rebuild()
    
    # Test
    try:
        response = await ai_logic_service.get_answer("Test question")
        print(f"✓ {provider_name} works! Response time: {response.processing_time}s")
    except Exception as e:
        print(f"✗ {provider_name} failed: {e}")

async def compare_providers():
    """Compare both providers."""
    await test_provider("Ollama")
    await test_provider("Groq")

# asyncio.run(compare_providers())


# ═══════════════════════════════════════════════════════════════════════════════════
# Configuration reference
# ═══════════════════════════════════════════════════════════════════════════════════

"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ LLM Provider Configuration Reference                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ LOCAL OLLAMA:                                                               │
│   LLM_PROVIDER=local                                                        │
│   OLLAMA_BASE_URL=http://localhost:11434                                    │
│   OLLAMA_MODEL_NAME=llama3.2:3b  (or mistral, neural-chat, etc.)           │
│                                                                             │
│ GROQ CLOUD API:                                                             │
│   LLM_PROVIDER=groq                                                         │
│   GROQ_API_KEY=gsk_xxxxxxxxxxxxx                                            │
│   GROQ_MODEL_NAME=mixtral-8x7b-32768  (or llama2-70b-4096, etc.)           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

OLLAMA MODELS:
  • llama3.2:3b     - Fast, low resource (default)
  • mistral         - Powerful general purpose
  • neural-chat     - Optimized for conversations
  • llama2          - Popular, stable
  • dolphin-phi     - Lightweight alternative
  • gemma           - Google's model

GROQ MODELS:
  • mixtral-8x7b-32768      - Fastest (default)
  • llama2-70b-4096         - Most powerful
  • gemma-7b-it             - Lightweight
  • llama-3-70b-versatile   - Latest, versatile
"""

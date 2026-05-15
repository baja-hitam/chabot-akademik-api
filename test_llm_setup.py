"""
Test script to verify both LLM providers work correctly.
Run this to validate your setup before using the application.

Usage:
    python test_llm_setup.py
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from app.core.config import get_settings
from app.services.ai_logic import ai_logic_service


async def test_llm_provider():
    """Test the configured LLM provider."""
    settings = get_settings()
    
    print("\n" + "="*70)
    print("LLM Provider Configuration Test")
    print("="*70)
    
    # Show current configuration
    print(f"\n📋 Configuration:")
    print(f"  LLM_PROVIDER: {settings.LLM_PROVIDER}")
    
    if settings.LLM_PROVIDER.lower() == "groq":
        print(f"  GROQ_API_KEY: {'✓ Set' if settings.GROQ_API_KEY else '✗ Not set'}")
        print(f"  GROQ_MODEL: {settings.GROQ_MODEL_NAME}")
    else:
        print(f"  OLLAMA_BASE_URL: {settings.OLLAMA_BASE_URL}")
        print(f"  OLLAMA_MODEL: {settings.OLLAMA_MODEL_NAME}")
    
    # Test LLM connection
    print(f"\n🔌 Testing LLM Connection...")
    try:
        llm = ai_logic_service._get_llm()
        print(f"  ✓ LLM initialized successfully")
        print(f"  Provider type: {type(llm).__name__}")
        
        # Test health check
        if ai_logic_service.is_healthy():
            print(f"  ✓ LLM health check passed")
        else:
            print(f"  ✗ LLM health check failed")
            return False
            
    except Exception as e:
        print(f"  ✗ Failed to initialize LLM: {e}")
        return False
    
    # Test simple query
    print(f"\n💬 Testing Simple Query...")
    try:
        test_question = "Apa itu akademik?"
        response = await ai_logic_service.get_answer(test_question)
        
        print(f"  Question: {test_question}")
        print(f"  Answer: {response.answer[:100]}...")
        print(f"  Processing time: {response.processing_time}s")
        print(f"  ✓ Query test passed")
        
    except Exception as e:
        print(f"  ✗ Query test failed: {e}")
        return False
    
    print("\n" + "="*70)
    print("✅ All tests passed! Your LLM setup is working correctly.")
    print("="*70 + "\n")
    return True


async def test_streaming():
    """Test streaming response."""
    print("\n" + "="*70)
    print("Streaming Response Test")
    print("="*70)
    
    print(f"\n🌊 Testing Streaming Response...")
    try:
        test_question = "Berikan penjelasan singkat tentang akademik"
        print(f"  Question: {test_question}")
        print(f"  Response: ", end="", flush=True)
        
        async for chunk in ai_logic_service.stream_answer(test_question):
            print(chunk, end="", flush=True)
        
        print(f"\n  ✓ Streaming test passed")
        
    except Exception as e:
        print(f"\n  ✗ Streaming test failed: {e}")
        return False
    
    print("="*70 + "\n")
    return True


async def main():
    """Run all tests."""
    provider_ok = await test_llm_provider()
    
    if provider_ok:
        # Only test streaming if basic test passed
        await test_streaming()


if __name__ == "__main__":
    # Check .env file exists
    if not os.path.exists(".env"):
        print("\n⚠️  WARNING: .env file not found!")
        print("Please copy .env.example to .env and update with your configuration:")
        print("  cp .env.example .env")
        print("  # Edit .env with your LLM provider settings")
        exit(1)
    
    asyncio.run(main())

import asyncio
from app.services.vector_store import vector_store_service

def test_reranker():
    print("Testing BGE Reranker Pipeline...")
    # This will load both BGE-M3 (embeddings) and BGE-Reranker-V2-M3
    results = vector_store_service.search_similar(
        query="jelaskan topik apa saja yang ada di kkp",
        top_k=5, 
        category="panduan_topik_kkp" # Try with or without category
    )
    
    print(f"Returned {len(results)} final documents.")
    for i, doc in enumerate(results):
        print(f"[{i+1}] Score: {doc['relevance_score']:.4f} | Source: {doc['source']}")

if __name__ == "__main__":
    test_reranker()

"""
ChromaDB Repository - Data Access Layer
Handles all direct interactions with the ChromaDB vector database,
including collection management, document insertion, and similarity queries.
"""

import logging
from typing import Any

import chromadb

from app.core.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


class ChromaRepository:
    """Wrapper around ChromaDB for academic document storage and retrieval."""

    def __init__(self) -> None:
        self._client: chromadb.ClientAPI | None = None
        self._collection = None

    # ── Initialization ────────────────────────────────────────────

    def _get_client(self) -> chromadb.ClientAPI:
        """Lazy-initialize the ChromaDB persistent client."""
        if self._client is None:
            persist_dir = str(settings.chroma_persist_path)
            logger.info("Initializing ChromaDB client at: %s", persist_dir)
            self._client = chromadb.PersistentClient(path=persist_dir)
        return self._client

    def _get_collection(self):
        """Get or create the academic documents collection."""
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION_NAME,
                metadata={"description": "Academic documents for RAG chatbot"},
            )
            logger.info(
                "Collection '%s' ready — %d documents",
                settings.CHROMA_COLLECTION_NAME,
                self._collection.count(),
            )
        return self._collection

    # ── CRUD Operations ───────────────────────────────────────────

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> int:
        """
        Add documents with pre-computed embeddings to the collection.

        Args:
            ids: Unique identifiers for each document chunk.
            documents: Raw text content of each chunk.
            metadatas: Metadata dicts associated with each chunk.
            embeddings: Pre-computed embedding vectors for each chunk.

        Returns:
            Number of documents added.
        """
        collection = self._get_collection()
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        logger.info("Added %d documents to collection", len(ids))
        return len(ids)

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Query the collection for similar documents.

        Args:
            query_embedding: The embedding vector of the query.
            n_results: Number of results to return.
            where: Optional metadata filter (e.g., {"category": "kurikulum"}).

        Returns:
            ChromaDB query results dict with documents, metadatas, distances.
        """
        collection = self._get_collection()
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = collection.query(**kwargs)
        logger.debug("Query returned %d results", len(results.get("ids", [[]])[0]))
        return results

    def get_collection_info(self) -> dict[str, Any]:
        """
        Return summary metadata about the current collection.

        Returns:
            Dict containing collection name, total document count,
            and a sorted list of distinct category values.
        """
        collection = self._get_collection()
        count = collection.count()

        # Retrieve unique categories
        categories: list[str] = []
        if count > 0:
            all_data = collection.get(include=["metadatas"])
            if all_data["metadatas"]:
                cats = {
                    m.get("category", "lainnya") for m in all_data["metadatas"] if m
                }
                categories = sorted(cats)

        return {
            "collection_name": settings.CHROMA_COLLECTION_NAME,
            "document_count": count,
            "categories": categories,
        }

    def delete_by_source(self, source: str) -> int:
        """
        Delete all chunks that belong to a specific source file.

        Args:
            source: The source filename to delete (matched against the
                    ``source`` metadata field).

        Returns:
            Number of chunks deleted.
        """
        collection = self._get_collection()
        existing = collection.get(where={"source": source}, include=[])
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
            logger.info(
                "Deleted %d chunks from source: %s", len(existing["ids"]), source
            )
            return len(existing["ids"])
        return 0

    def get_sources_by_base_name(self, base_name: str) -> list[str]:
        """
        Return all distinct source filenames that share the same
        normalized document base name (e.g. ``'panduan_krs'``).

        Args:
            base_name: Normalized document base name (without year/extension).

        Returns:
            List of source filenames stored in the collection.
        """
        collection = self._get_collection()
        if collection.count() == 0:
            return []

        all_data = collection.get(include=["metadatas"])
        sources: set[str] = set()
        for meta in all_data.get("metadatas") or []:
            if meta and meta.get("document_base_name") == base_name:
                src = meta.get("source", "")
                if src:
                    sources.add(src)
        return list(sources)

    def get_ingested_files(self) -> list[dict[str, Any]]:
        """
        Return a list of unique ingested files and their metadata.
        """
        collection = self._get_collection()
        if collection.count() == 0:
            return []

        all_data = collection.get(include=["metadatas"])
        files_map = {}
        for meta in (all_data.get("metadatas") or []):
            if not meta:
                continue
            source = meta.get("source")
            if source and source not in files_map:
                files_map[source] = {
                    "filename": source,
                    "kd_prodi": meta.get("kd_prodi"),
                    "category": meta.get("category"),
                    "document_year": meta.get("document_year"),
                    "is_latest": meta.get("is_latest")
                }
        return list(files_map.values())

    def update_metadata_by_source(
        self, source: str, metadata_update: dict[str, Any]
    ) -> int:
        """
        Merge ``metadata_update`` into the existing metadata for every chunk
        that belongs to ``source``.

        Args:
            source: The source filename whose chunks should be updated.
            metadata_update: Key/value pairs to merge into each chunk's metadata.

        Returns:
            Number of chunks updated.
        """
        collection = self._get_collection()
        existing = collection.get(where={"source": source}, include=["metadatas"])
        if not existing["ids"]:
            return 0

        merged_metadatas = [
            {**meta, **metadata_update}
            for meta in (existing["metadatas"] or [{}] * len(existing["ids"]))
        ]
        collection.update(ids=existing["ids"], metadatas=merged_metadatas)
        logger.info(
            "Updated metadata for %d chunks of source '%s': %s",
            len(existing["ids"]),
            source,
            metadata_update,
        )
        return len(existing["ids"])

    def reset_collection(self) -> None:
        """
        Delete and recreate the collection.

        .. warning::
            This permanently removes **all** stored documents. Use with caution.
        """
        client = self._get_client()
        try:
            client.delete_collection(settings.CHROMA_COLLECTION_NAME)
            logger.warning("Collection '%s' deleted", settings.CHROMA_COLLECTION_NAME)
        except Exception:
            pass
        self._collection = None
        self._get_collection()


# ── Singleton instance ────────────────────────────────────────────
chroma_repo = ChromaRepository()

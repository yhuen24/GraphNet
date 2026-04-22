"""
ai/chunk_store.py — Document chunk storage for GraphNet.

Stores the original text chunks from processed documents so they can be
retrieved at query time and fed to the LLM as reasoning context.

This is what enables the "smart answering" upgrade: instead of the LLM
only seeing entity names and relationship labels, it can read the actual
source text and make judgments, comparisons, and decisions.

Storage backend: ChromaDB (separate collection from the entity embeddings).
"""

import logging
import hashlib
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class ChunkStore:
    """
    Stores and retrieves document text chunks using ChromaDB embeddings.

    Each entry stores:
        - id        : deterministic hash of (source + chunk_index)
        - document  : the actual text chunk
        - metadata  : {"source": filename, "chunk_index": int, "char_count": int}
        - embedding : vector from Google Gemini or TF-IDF fallback

    At query time, the QueryAgent calls `search()` to find the most relevant
    chunks and passes them to the LLM alongside the graph data.
    """

    COLLECTION_NAME = "graphnet_chunks"

    def __init__(self, chroma_dir: str = "graphnet_chroma_db"):
        self._chroma_dir = chroma_dir
        self._client = None
        self._collection = None
        self._embed_fn = None
        self._mode = "none"

    # ── Initialisation ────────────────────────────────────────────────────────

    def initialize(self, google_api_key: str = "") -> bool:
        """
        Set up embedding function and connect to ChromaDB.
        Should be called after EmbeddingStore.initialize() so we reuse
        the same ChromaDB directory.
        """
        # 1. Set up embedding backend
        if google_api_key:
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=google_api_key)

                def _google_embed(texts: List[str]) -> List[List[float]]:
                    all_embs = []
                    for i in range(0, len(texts), 100):
                        batch = texts[i:i + 100]
                        result = client.models.embed_content(
                            model="gemini-embedding-001",
                            contents=batch,
                            config=types.EmbedContentConfig(
                                task_type="RETRIEVAL_DOCUMENT",
                            ),
                        )
                        for emb in result.embeddings:
                            all_embs.append(emb.values)
                    return all_embs

                self._embed_fn = _google_embed
                self._mode = "google"
                logger.info("ChunkStore: using Google gemini-embedding-001")
            except Exception as e:
                logger.warning(f"Google embedding unavailable for chunks ({e})")

        # 2. TF-IDF fallback
        if self._embed_fn is None:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                self._tfidf = TfidfVectorizer(
                    analyzer="word", ngram_range=(1, 2), max_features=768
                )
                self._tfidf_fitted = False
                self._mode = "tfidf"

                def _tfidf_embed(texts: List[str]) -> List[List[float]]:
                    if not self._tfidf_fitted:
                        # Fit on existing + new texts
                        existing = self._all_texts()
                        corpus = existing + texts if existing else texts
                        if corpus:
                            self._tfidf.fit(corpus)
                            self._tfidf_fitted = True
                    vecs = self._tfidf.transform(texts).toarray()
                    return vecs.tolist()

                self._embed_fn = _tfidf_embed
                logger.info("ChunkStore: using TF-IDF fallback")
            except ImportError:
                logger.warning("scikit-learn not available for chunk embeddings")
                self._mode = "none"
                return False

        # 3. Connect to ChromaDB
        try:
            import chromadb

            self._client = chromadb.PersistentClient(path=self._chroma_dir)
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"ChunkStore: ChromaDB ready ({self._collection.count()} chunks stored)"
            )
            return True

        except ImportError:
            logger.error("chromadb not installed — chunk store disabled")
            return False
        except Exception as e:
            logger.error(f"ChunkStore ChromaDB init failed: {e}")
            return False

    # ── Chunk ID generation ───────────────────────────────────────────────────

    @staticmethod
    def _make_chunk_id(source: str, chunk_index: int) -> str:
        """Deterministic ID so re-processing the same doc doesn't duplicate."""
        raw = f"{source}::chunk::{chunk_index}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # ── Storage ───────────────────────────────────────────────────────────────

    def store_chunks(self, chunks: List[str], source: str) -> int:
        """
        Store text chunks from a processed document.

        Args:
            chunks: List of text chunks from DocumentProcessor.chunk_text()
            source: The filename / document identifier

        Returns:
            Number of new chunks stored
        """
        if self._collection is None or not chunks:
            return 0

        # Check which chunks are already stored
        chunk_ids = [self._make_chunk_id(source, i) for i in range(len(chunks))]
        try:
            existing = self._collection.get(ids=chunk_ids, include=[])
            existing_ids = set(existing["ids"])
        except Exception:
            existing_ids = set()

        # Filter to new chunks only
        new_ids = []
        new_texts = []
        new_metas = []
        for i, (cid, chunk) in enumerate(zip(chunk_ids, chunks)):
            if cid not in existing_ids:
                new_ids.append(cid)
                new_texts.append(chunk)
                new_metas.append({
                    "source": source,
                    "chunk_index": i,
                    "char_count": len(chunk),
                })

        if not new_ids:
            logger.info(f"ChunkStore: all {len(chunks)} chunks from '{source}' already stored")
            return 0

        try:
            # Embed in batches (chunks can be large)
            embeddings = []
            batch_size = 10  # Smaller batches for large text chunks
            for i in range(0, len(new_texts), batch_size):
                batch = new_texts[i:i + batch_size]
                # Truncate very long chunks for embedding (keep full text in doc)
                truncated = [t[:8000] for t in batch]
                embeddings.extend(self._embed_fn(truncated))

            self._collection.add(
                ids=new_ids,
                embeddings=embeddings,
                documents=new_texts,
                metadatas=new_metas,
            )
            logger.info(f"ChunkStore: stored {len(new_ids)} chunks from '{source}'")
            return len(new_ids)

        except Exception as e:
            logger.error(f"ChunkStore: failed to store chunks from '{source}': {e}")
            return 0

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5,
               source_filter: str = None) -> List[Dict[str, Any]]:
        """
        Find the most relevant document chunks for a query.

        Args:
            query:         Natural language query
            top_k:         Max chunks to return
            source_filter: Optional — restrict to chunks from this document

        Returns:
            List of dicts with keys: text, source, chunk_index, score
        """
        if self._collection is None or self._collection.count() == 0:
            return []

        try:
            # Truncate query for embedding
            q_emb = self._embed_fn([query[:4000]])[0]

            where_filter = None
            if source_filter:
                where_filter = {"source": source_filter}

            results = self._collection.query(
                query_embeddings=[q_emb],
                n_results=min(top_k, self._collection.count()),
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            output = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                score = 1.0 - (dist / 2.0)  # cosine distance → similarity
                output.append({
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "chunk_index": meta.get("chunk_index", 0),
                    "score": round(score, 4),
                })

            return output

        except Exception as e:
            logger.error(f"ChunkStore search failed: {e}")
            return []

    def get_chunks_by_source(self, source: str,
                             max_chunks: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks from a specific document, in order.

        Args:
            source: The filename / document identifier
            max_chunks: Safety limit

        Returns:
            List of chunk dicts sorted by chunk_index
        """
        if self._collection is None or self._collection.count() == 0:
            return []

        try:
            results = self._collection.get(
                where={"source": source},
                include=["documents", "metadatas"],
                limit=max_chunks,
            )

            chunks = []
            for doc, meta in zip(results["documents"], results["metadatas"]):
                chunks.append({
                    "text": doc,
                    "source": meta.get("source", source),
                    "chunk_index": meta.get("chunk_index", 0),
                })

            # Sort by chunk_index for reading order
            chunks.sort(key=lambda c: c["chunk_index"])
            return chunks

        except Exception as e:
            logger.error(f"ChunkStore get_chunks_by_source failed: {e}")
            return []

    def list_sources(self) -> List[str]:
        """Return a list of all document sources stored."""
        if self._collection is None or self._collection.count() == 0:
            return []
        try:
            results = self._collection.get(include=["metadatas"])
            sources = set()
            for meta in results["metadatas"]:
                src = meta.get("source", "")
                if src:
                    sources.add(src)
            return sorted(sources)
        except Exception:
            return []

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _all_texts(self) -> List[str]:
        if self._collection is None or self._collection.count() == 0:
            return []
        try:
            result = self._collection.get(include=["documents"])
            return result["documents"] or []
        except Exception:
            return []

    def count(self) -> int:
        if self._collection is None:
            return 0
        return self._collection.count()

    def clear(self) -> None:
        """Remove all stored chunks."""
        if self._client is None:
            return
        try:
            self._client.delete_collection(self.COLLECTION_NAME)
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("ChunkStore: cleared all chunks")
        except Exception as e:
            logger.error(f"ChunkStore clear failed: {e}")

    def save(self) -> None:
        """No-op — ChromaDB auto-persists."""
        pass
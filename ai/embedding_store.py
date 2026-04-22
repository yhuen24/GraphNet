"""
ai/embedding_store.py — Semantic embedding index for GraphNet.

Embeds entity names + descriptions using Google's Gemini embedding API so that
user queries like "dangote" match "Dangote Group" even with typos or
different phrasing.  Uses the new google.genai SDK with gemini-embedding-001.

Storage backend: ChromaDB (persistent local vector database).
Falls back to TF-IDF (scikit-learn) if the embedding API is unavailable.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """
    Manages a persistent ChromaDB vector index over graph entities.

    Each entry stores:
        - id       : node_id  (e.g. "organization:dangote group")
        - embedding: List[float] from Gemini or TF-IDF
        - document : the text that was embedded
        - metadata : {"node_id": ..., "text": ...}

    Public API is identical to the old JSON-based store so nothing else
    in the codebase needs to change.
    """

    COLLECTION_NAME = "graphnet_entities"

    def __init__(self, persist_file: str = "graphnet_embeddings.json"):
        # persist_file is kept as a parameter for API compatibility but is
        # no longer used — ChromaDB manages its own directory.
        self._chroma_dir = "graphnet_chroma_db"
        self._client = None
        self._collection = None
        self._embed_fn = None
        self._mode = "none"

        # TF-IDF fallback state
        self._tfidf = None
        self._tfidf_fitted = False

    # ── Initialisation ────────────────────────────────────────────────────────

    def initialize(self, google_api_key: str = "") -> bool:
        """
        Set up the embedding function and connect to ChromaDB.
        Tries Google first, then TF-IDF.
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

                # Quick smoke test
                _google_embed(["test"])
                self._embed_fn = _google_embed
                self._mode = "google"
                logger.info("EmbeddingStore: using Google gemini-embedding-001")
            except Exception as e:
                logger.warning(f"Google embedding unavailable ({e}), trying TF-IDF fallback")

        # 2. TF-IDF fallback
        if self._embed_fn is None:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                self._tfidf = TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(2, 4), max_features=512
                )
                self._tfidf_fitted = False
                self._mode = "tfidf"
                logger.info("EmbeddingStore: using TF-IDF fallback")
            except ImportError:
                logger.warning("scikit-learn not available; embedding store disabled")
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
                f"EmbeddingStore: ChromaDB ready at '{self._chroma_dir}' "
                f"({self._collection.count()} entities loaded)"
            )
            return True

        except ImportError:
            logger.error("chromadb is not installed. Run: pip install chromadb")
            return False
        except Exception as e:
            logger.error(f"ChromaDB init failed: {e}")
            return False

    # ── Embedding helpers ─────────────────────────────────────────────────────

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts using the active backend."""
        if self._mode == "google":
            return self._embed_fn(texts)
        elif self._mode == "tfidf":
            return self._tfidf_embed(texts)
        else:
            return [[0.0] * 64 for _ in texts]

    def _tfidf_embed(self, texts: List[str]) -> List[List[float]]:
        """Embed using the TF-IDF vectoriser."""
        if not self._tfidf_fitted:
            existing = self._all_texts()
            corpus = existing + texts if existing else texts
            self._tfidf.fit(corpus)
            self._tfidf_fitted = True

            # Re-embed everything already stored so vectors stay consistent
            if existing:
                ids = self._all_ids()
                new_vecs = self._tfidf.transform(existing).toarray().tolist()
                self._collection.upsert(ids=ids, embeddings=new_vecs)

        vecs = self._tfidf.transform(texts).toarray()
        return vecs.tolist()

    # ── ChromaDB helpers ──────────────────────────────────────────────────────

    def _all_ids(self) -> List[str]:
        """Return all stored IDs (node_ids)."""
        if self._collection.count() == 0:
            return []
        result = self._collection.get(include=[])
        return result["ids"]

    def _all_texts(self) -> List[str]:
        """Return all stored document texts."""
        if self._collection.count() == 0:
            return []
        result = self._collection.get(include=["documents"])
        return result["documents"] or []

    @property
    def entries(self) -> List[Dict[str, Any]]:
        """
        Compatibility shim — exposes indexed entries in the old format
        used by `main._reindex_existing_entities()`.

        Returns: [{"node_id": str, "text": str}, ...]
        """
        if self._collection is None or self._collection.count() == 0:
            return []
        result = self._collection.get(include=["documents"])
        return [
            {"node_id": id_, "text": doc}
            for id_, doc in zip(result["ids"], result["documents"])
        ]

    # ── Public API ────────────────────────────────────────────────────────────

    def add_entity(self, node_id: str, name: str, entity_type: str = "",
                   description: str = "") -> None:
        """
        Add or update an entity's embedding.
        The text embedded is: "{type}: {name}. {description}"
        """
        if self._collection is None:
            return

        # Skip if already indexed
        existing = self._collection.get(ids=[node_id], include=[])
        if existing["ids"]:
            return

        text = f"{entity_type}: {name}"
        if description:
            text += f". {description}"

        try:
            emb = self._embed([text])[0]
            self._collection.add(
                ids=[node_id],
                embeddings=[emb],
                documents=[text],
                metadatas=[{"node_id": node_id, "text": text}],
            )
        except Exception as e:
            logger.error(f"Failed to embed entity {node_id}: {e}")

    def add_entities_batch(self, entities: List[Dict[str, str]]) -> None:
        """
        Batch-add entities. Each dict must have: node_id, name.
        Optional: entity_type, description.
        """
        if self._collection is None or not entities:
            return

        # Filter out already-indexed entities
        all_ids = set(self._all_ids())
        new_entities = [e for e in entities if e["node_id"] not in all_ids]
        if not new_entities:
            return

        texts = []
        for ent in new_entities:
            t = f"{ent.get('entity_type', '')}: {ent['name']}"
            if ent.get("description"):
                t += f". {ent['description']}"
            texts.append(t)

        try:
            embeddings = self._embed(texts)
            self._collection.add(
                ids=[e["node_id"] for e in new_entities],
                embeddings=embeddings,
                documents=texts,
                metadatas=[
                    {"node_id": e["node_id"], "text": t}
                    for e, t in zip(new_entities, texts)
                ],
            )
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")

    def search(self, query: str, top_k: int = 10,
               threshold: float = 0.25) -> List[Tuple[str, float]]:
        """
        Semantic search: return (node_id, score) pairs sorted best-first.

        Args:
            query:     Natural language query string.
            top_k:     Max results.
            threshold: Minimum cosine similarity to include.

        Returns:
            List of (node_id, similarity_score) tuples.
        """
        if self._collection is None or self._collection.count() == 0:
            return []

        try:
            q_emb = self._embed([query])[0]

            results = self._collection.query(
                query_embeddings=[q_emb],
                n_results=min(top_k, self._collection.count()),
                include=["metadatas", "distances"],
            )

            output = []
            for node_id, distance in zip(
                results["ids"][0], results["distances"][0]
            ):
                # ChromaDB cosine space returns distance (0=identical, 2=opposite)
                # Convert to similarity score in [0, 1]
                score = 1.0 - (distance / 2.0)
                if score >= threshold:
                    output.append((node_id, round(score, 4)))

            return output

        except Exception as e:
            logger.error(f"Embedding search failed: {e}")
            return []

    def save(self) -> None:
        """
        No-op — ChromaDB persists automatically on every write.
        Kept for API compatibility with the rest of the codebase.
        """
        logger.debug("EmbeddingStore.save() called — ChromaDB auto-persists, nothing to do.")

    def clear(self) -> None:
        """Remove all embeddings from the collection."""
        if self._client is None:
            return
        try:
            self._client.delete_collection(self.COLLECTION_NAME)
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            self._tfidf_fitted = False
            logger.info("EmbeddingStore: collection cleared.")
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")

    def count(self) -> int:
        """Return the number of indexed entities."""
        if self._collection is None:
            return 0
        return self._collection.count()
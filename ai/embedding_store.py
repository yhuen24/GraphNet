"""
ai/embedding_store.py — Semantic embedding index for GraphNet.

Embeds entity names + descriptions using Google's Gemini embedding API so that
user queries like "dangote" match "Dangote Group" even with typos or
different phrasing.  Uses the new google.genai SDK with gemini-embedding-001.

Falls back to TF-IDF (scikit-learn) if embedding API is unavailable.
"""

import logging
import json
import os
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


# ── Vector math helpers (no heavy deps) ──────────────────────────────────────

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


class EmbeddingStore:
    """
    Manages a lightweight in-memory vector index over graph entities.

    Each entry is:
        { "node_id": str, "text": str, "embedding": List[float] }

    The store tries Google's Generative AI embedding model first; if that
    fails (no key, network error) it falls back to a local TF-IDF approach
    so the system always works.
    """

    def __init__(self, persist_file: str = "graphnet_embeddings.json"):
        self.persist_file = persist_file
        self.entries: List[Dict[str, Any]] = []  # {"node_id", "text", "embedding"}
        self._matrix: Optional[np.ndarray] = None  # cached N×D matrix
        self._embed_fn = None  # callable: List[str] → List[List[float]]
        self._mode = "none"
        self._dirty = False

    # ── Initialisation ───────────────────────────────────────────────────────

    def initialize(self, google_api_key: str = "") -> bool:
        """
        Set up the embedding function.  Tries Google first, then TF-IDF.
        """
        # Try Google embedding (new google.genai SDK)
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

        # Fallback — TF-IDF via scikit-learn
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

        # Load persisted embeddings
        self._load()
        return True

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        if not os.path.exists(self.persist_file):
            return
        try:
            with open(self.persist_file, "r") as f:
                data = json.load(f)
            self.entries = data.get("entries", [])
            self._rebuild_matrix()
            logger.info(f"Loaded {len(self.entries)} embeddings from {self.persist_file}")
        except Exception as e:
            logger.warning(f"Could not load embeddings: {e}")

    def save(self):
        """Persist embeddings to disk."""
        if not self._dirty:
            return
        try:
            with open(self.persist_file, "w") as f:
                json.dump({"entries": self.entries}, f)
            self._dirty = False
            logger.info(f"Saved {len(self.entries)} embeddings to {self.persist_file}")
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")

    def _rebuild_matrix(self):
        """Rebuild the numpy matrix from self.entries for fast search."""
        if self.entries:
            self._matrix = np.array([e["embedding"] for e in self.entries], dtype=np.float32)
        else:
            self._matrix = None

    # ── Core operations ──────────────────────────────────────────────────────

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts using the active backend."""
        if self._mode == "google":
            return self._embed_fn(texts)
        elif self._mode == "tfidf":
            return self._tfidf_embed(texts)
        else:
            # No backend — return zero vectors
            return [[0.0] * 64 for _ in texts]

    def _tfidf_embed(self, texts: List[str]) -> List[List[float]]:
        """Embed using the TF-IDF vectoriser."""
        if not self._tfidf_fitted and self.entries:
            corpus = [e["text"] for e in self.entries]
            self._tfidf.fit(corpus + texts)
            self._tfidf_fitted = True
            # Re-embed everything with the new vocabulary
            all_texts = [e["text"] for e in self.entries]
            if all_texts:
                vecs = self._tfidf.transform(all_texts).toarray()
                for i, entry in enumerate(self.entries):
                    entry["embedding"] = vecs[i].tolist()
                self._rebuild_matrix()
                self._dirty = True

        if not self._tfidf_fitted:
            self._tfidf.fit(texts)
            self._tfidf_fitted = True

        vecs = self._tfidf.transform(texts).toarray()
        return vecs.tolist()

    def add_entity(self, node_id: str, name: str, entity_type: str = "",
                   description: str = "") -> None:
        """
        Add or update an entity's embedding.

        The text embedded is: "{type}: {name}. {description}"
        """
        # Check if already present
        for entry in self.entries:
            if entry["node_id"] == node_id:
                return  # Already indexed

        text = f"{entity_type}: {name}"
        if description:
            text += f". {description}"

        try:
            emb = self._embed([text])[0]
            self.entries.append({
                "node_id": node_id,
                "text": text,
                "embedding": emb,
            })
            self._dirty = True
            self._rebuild_matrix()
        except Exception as e:
            logger.error(f"Failed to embed entity {node_id}: {e}")

    def add_entities_batch(self, entities: List[Dict[str, str]]) -> None:
        """
        Batch-add entities.  Each dict must have: node_id, name.
        Optional: entity_type, description.
        """
        existing_ids = {e["node_id"] for e in self.entries}
        new_entities = [e for e in entities if e["node_id"] not in existing_ids]
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
            for ent, emb in zip(new_entities, embeddings):
                self.entries.append({
                    "node_id": ent["node_id"],
                    "text": texts[new_entities.index(ent)],
                    "embedding": emb,
                })
            self._dirty = True
            self._rebuild_matrix()
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
        if not self.entries or self._matrix is None:
            return []

        try:
            q_emb = np.array(self._embed([query])[0], dtype=np.float32)
            # Cosine similarity against the whole matrix
            norms = np.linalg.norm(self._matrix, axis=1)
            q_norm = np.linalg.norm(q_emb)
            if q_norm == 0:
                return []
            sims = self._matrix @ q_emb / (norms * q_norm + 1e-10)

            # Rank
            indices = np.argsort(-sims)
            results = []
            for idx in indices[:top_k]:
                score = float(sims[idx])
                if score < threshold:
                    break
                results.append((self.entries[idx]["node_id"], score))
            return results

        except Exception as e:
            logger.error(f"Embedding search failed: {e}")
            return []

    def clear(self):
        """Remove all embeddings."""
        self.entries = []
        self._matrix = None
        self._dirty = True
        self.save()
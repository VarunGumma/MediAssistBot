"""
FAISS Index Loader Utility

This module provides utilities to load pre-generated FAISS indices
for use in the assistant application.

Usage in your application:
    from faiss_loader import FAISSLoader

    # Load pre-generated index
    loader = FAISSLoader("./faiss_db")

    # Search
    results = loader.search("patient with fever", top_k=5)

    # Get document by index
    doc = loader.get_document(0)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch



@dataclass
class SearchResult:
    """A single search result from FAISS."""

    content: str
    source: str
    score: float
    index: int
    metadata: dict[str, Any]


class FAISSLoader:
    def __init__(self, index_dir: str | Path):
        self.index_dir = Path(index_dir).expanduser()
        self._validate_index_dir()
        with open(self.index_dir / "index_metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        if "provider" not in self.metadata:
            self.metadata["provider"] = "huggingface"
        self._load_index()
        self._load_documents()

    def _validate_index_dir(self) -> None:
        for f in [
            "index.faiss",
            "embeddings.npy",
            "documents.json",
            "index_metadata.json",
        ]:
            if not (self.index_dir / f).exists():
                raise FileNotFoundError(
                    f"Required file not found: {self.index_dir / f}\nGenerate the index with: python generate_faiss_embeddings.py"
                )

    def _load_index(self) -> None:
        self.index = faiss.read_index(str(self.index_dir / "index.faiss"))
        self.embeddings = np.load(self.index_dir / "embeddings.npy").astype("float32")

    def _load_documents(self) -> None:
        with open(self.index_dir / "documents.json", "r", encoding="utf-8") as f:
            self.documents = json.load(f)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if top_k <= 0:
            return []
        sims = np.dot(self.embeddings, query_embedding.T)[:, 0]
        if (top_k := min(top_k, sims.shape[0])) == 0:
            return []
        top_idxs = np.argpartition(-sims, top_k - 1)[:top_k]
        sorted_idxs = top_idxs[np.argsort(-sims[top_idxs])]
        return [
            SearchResult(
                content=(d := self.documents[i])["page_content"],
                source=d["metadata"].get("source", "unknown"),
                score=float(sims[i]),
                index=int(i),
                metadata=d["metadata"],
            )
            for i in sorted_idxs
        ]

    def get_document(self, index: int) -> dict[str, Any]:
        if not 0 <= index < len(self.documents):
            raise IndexError(f"Document index out of range: {index}")
        return self.documents[index]

    def get_embedding(self, index: int) -> np.ndarray:
        if not 0 <= index < len(self.embeddings):
            raise IndexError(f"Embedding index out of range: {index}")
        return self.embeddings[index]

    @property
    def num_documents(self) -> int:
        return len(self.documents)

    @property
    def embedding_dimension(self) -> int:
        return self.embeddings.shape[1]

    @property
    def embedding_model(self) -> str:
        return self.metadata["embedding_model"]

    @property
    def provider(self) -> str:
        return self.metadata.get("provider", "huggingface")

    def info(self) -> dict[str, Any]:
        return {
            "num_documents": self.num_documents,
            "embedding_dimension": self.embedding_dimension,
            "embedding_model": self.embedding_model,
            "provider": self.provider,
            "created_at": self.metadata.get("created_at", "unknown"),
            "chunk_size": self.metadata.get("chunk_size"),
            "chunk_overlap": self.metadata.get("chunk_overlap"),
            "docs_dir": self.metadata.get("docs_dir"),
        }


class FAISSEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._init_hf()

    def _encode(
        self, texts: list[str], *, batch_size: int = 128
    ) -> np.ndarray:
        arr = np.asarray(
            self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True
            ),
            dtype="float32",
        )
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    def _init_hf(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            from src.config import CACHE_DIR
        except ImportError as exc:
            raise ImportError(
                f"Required libraries not installed: {exc}\nInstall with: pip install torch sentence-transformers"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA-capable GPU is required to initialize FAISS embeddings. No CPU fallback is available."
            )
        self.model = SentenceTransformer(
            self.model_name,
            device="cuda",
            cache_folder=str(CACHE_DIR / "sentence_transformers"),
            model_kwargs={"torch_dtype": torch.bfloat16},
        )

    def embed(self, text: str) -> np.ndarray:
        return self._encode([text], batch_size=1)[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, batch_size=512)
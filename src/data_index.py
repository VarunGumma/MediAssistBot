from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


SUPPORTED_MARKDOWN_SUFFIXES = {".md", ".markdown", ".mkd", ".mdown"}


def _scan_markdown_files(root: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_MARKDOWN_SUFFIXES
        ]
    )


def _fingerprint_files(paths: Sequence[Path]) -> str:
    hasher = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        hasher.update(str(path).encode("utf-8"))
        hasher.update(str(stat.st_size).encode("utf-8"))
        hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
    return hasher.hexdigest()


def _load_document(path: Path, base_dir: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    relative_path = path.relative_to(base_dir)
    return Document(page_content=text, metadata={"source": str(relative_path)})


@dataclass(frozen=True)
class KnowledgeBase:
    """Represents the prepared markdown corpus ready for vectorization."""

    documents: List[Document]
    fingerprint: str


def load_knowledge_base(
    docs_dir: Path, chunk_size: int, chunk_overlap: int
) -> KnowledgeBase:
    if not docs_dir.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {docs_dir}")
    if not (md_files := _scan_markdown_files(docs_dir)):
        return KnowledgeBase(documents=[], fingerprint="")
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    ).split_documents([_load_document(p, docs_dir) for p in md_files])
    for idx, chunk in enumerate(chunks):
        chunk.metadata.setdefault("chunk_index", idx)
    return KnowledgeBase(documents=chunks, fingerprint=_fingerprint_files(md_files))

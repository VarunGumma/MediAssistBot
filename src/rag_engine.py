from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import BOT_NAME, CACHE_DIR
from .data_index import KnowledgeBase, load_knowledge_base

device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class RetrievedChunk:
    """Represents a single chunk selected for the final answer."""

    content: str
    source: str
    score: float


@dataclass
class Retrieval:
    answer: str
    supporting_chunks: List[RetrievedChunk]
    usage: Optional[dict]


class RagEngine:
    """RAG pipeline powered by local Hugging Face embeddings and generation."""

    def __init__(
        self,
        provider: str,
        api_key: Optional[str],
        docs_dir,
        chunk_size: int,
        chunk_overlap: int,
        embedding_model: str,
        chat_model: str,
        top_k: int,
        generation_config: Dict[str, Any],
    ) -> None:
        if provider != "huggingface":
            raise ValueError("RagEngine now supports only the Hugging Face provider.")
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA-capable GPU is required for {BOT_NAME} embeddings and generation. No CPU fallback is available."
            )
        self._provider = provider
        self._api_key = api_key or ""
        self._docs_dir = Path(docs_dir).expanduser()
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._embedding_model = embedding_model
        self._chat_model = chat_model
        self._top_k = top_k
        self._generation_config = dict(generation_config or {})
        self._knowledge_base = None
        self._hf_index = None
        self._hf_doc_embeddings = None
        self._hf_tokenizer = None
        self._hf_model = None
        self._hf_embedder = None

    def _ensure_knowledge_base(self) -> KnowledgeBase:
        if not self._knowledge_base:
            self._knowledge_base = load_knowledge_base(
                self._docs_dir, self._chunk_size, self._chunk_overlap
            )
        return self._knowledge_base

    @staticmethod
    def _format_history(
        history: Optional[List[Dict[str, str]]], max_turns: int = 10
    ) -> str:
        if not history:
            return ""
        trimmed = history[-max_turns:]
        lines: List[str] = []
        for idx, message in enumerate(trimmed, 1):
            role = message["role"].lower()
            content = message["content"].strip()
            if not content:
                continue
            if role == "assistant":
                speaker = BOT_NAME
            elif role == "user":
                speaker = "USER"
            lines.append(f"{idx}. **{speaker}**: {content}")
        return "\n".join(lines)

    def _hf_cache_dir(self) -> Path:
        return (
            CACHE_DIR
            / f"hf_{self._ensure_knowledge_base().fingerprint}_{self._embedding_model.replace('/', '_').replace(':', '_')}"
        )

    def _get_sentence_embedder(self) -> SentenceTransformer:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA-capable GPU is required for local embeddings. No CPU fallback is available."
            )
        if not self._hf_embedder:
            model_kwargs = {
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "flash_attention_2",
            }
            tokenizer_kwargs = {"padding_side": "left"}
            try:
                self._hf_embedder = SentenceTransformer(
                    self._embedding_model,
                    device="cuda",
                    model_kwargs=model_kwargs,
                    tokenizer_kwargs=tokenizer_kwargs,
                )
            except TypeError as exc:
                raise RuntimeError(
                    "Installed sentence-transformers or the selected embedding model does not support FlashAttention 2. Update the dependency or choose a compatible model."
                ) from exc
        return self._hf_embedder

    def _hf_embed(
        self, texts: List[str], prompt_name: Optional[str] = None
    ) -> np.ndarray:
        model = self._get_sentence_embedder()
        encode_kwargs = {
            "batch_size": 384,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
            "show_progress_bar": True,
        }
        if prompt_name:
            encode_kwargs["prompt_name"] = prompt_name
        arr = np.asarray(model.encode(texts, **encode_kwargs), dtype="float32")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    def _ensure_hf_index(self) -> faiss.Index:
        if self._hf_index and self._hf_doc_embeddings is not None:
            return self._hf_index

        kb = self._ensure_knowledge_base()
        if not kb.documents:
            raise RuntimeError(
                "No markdown files were found in the knowledge base directory."
            )

        cache_dir = self._hf_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        emb_path, idx_path = cache_dir / "doc_embeddings.npy", cache_dir / "index.faiss"

        if emb_path.exists() and idx_path.exists():
            self._hf_doc_embeddings = np.load(emb_path).astype("float32")
            self._hf_index = faiss.read_index(str(idx_path))
        else:
            emb = self._hf_embed([doc.page_content for doc in kb.documents])
            idx = faiss.IndexFlatIP(emb.shape[1])
            idx.add(emb)
            np.save(emb_path, emb)
            faiss.write_index(idx, str(idx_path))
            self._hf_doc_embeddings, self._hf_index = emb, idx

        return self._hf_index

    def _ensure_hf_llm(self) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU is required for Hugging Face generation.")

        if not self._hf_tokenizer:
            tok = AutoTokenizer.from_pretrained(
                self._chat_model, trust_remote_code=True
            )
            if tok.pad_token_id is None and tok.eos_token:
                tok.pad_token = tok.eos_token
            self._hf_tokenizer = tok

        if not self._hf_model:
            torch.cuda.empty_cache()
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    self._chat_model,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                ).to(device)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load the Hugging Face chat model with FlashAttention 2. Verify flash-attn support and install the required CUDA extensions."
                ) from exc
            model.eval()
            self._hf_model = model

        return self._hf_tokenizer, self._hf_model

    def _generate_hf_response(
        self, messages: List[Dict[str, str]]
    ) -> str:
        tok, model = self._ensure_hf_llm()
        prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tok(prompt, return_tensors="pt", padding=False).to(model.device)
        gen_kwargs = {**self._generation_config}
        gen_kwargs["eos_token_id"] = tok.eos_token_id
        gen_kwargs["pad_token_id"] = tok.pad_token_id

        with torch.inference_mode():
            gen = model.generate(**inputs, **gen_kwargs)
        return tok.decode(
            gen[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

    def _build_retrieval_query(
        self,
        gender: str,
        age: int,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]],
    ) -> str:
        history_block = self._format_history(conversation_history)
        history_section = (
            f"**Conversation so far**:\n{history_block}\n\n" if history_block else ""
        )
        return (
            f"**Patient demographics**: gender=`{gender}`, age=`{age}`.\n{history_section}"
            f"**Latest user question**: {query.strip()}"
        )

    def _assemble_context(
        self,
        gender: str,
        age: int,
        query: str,
        chunks: List[RetrievedChunk],
    ) -> str:
        ctx = (
            "No supporting knowledge base excerpts were retrieved."
            if not chunks
            else "\n\n".join(
                [
                    "\n".join(
                        [
                            f"\t[{i}] **Source**: {c.source}",
                            "```",
                            c.content.strip(),
                            "```",
                        ]
                    )
                    for i, c in enumerate(chunks, 1)
                ]
            )
        )
        return (
            f" ► **Patient profile**:\nGender: `{gender}`, Age: `{age}`\n\n"
            f" ► **User question**:\n{query.strip()}\n\n"
            f" ► **Knowledge base excerpts**:\n{ctx}"
        )

    def _build_generation_messages(
        self,
        system_prompt: str,
        gender: str,
        age: int,
        query: str,
        chunks: List[RetrievedChunk],
        conversation_history: Optional[List[Dict[str, str]]],
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt.strip()},
        ]
        if conversation_history:
            for message in conversation_history[-10:]:
                role = message["role"].lower()
                if role not in {"user", "assistant"}:
                    continue
                content = message["content"].strip()
                if content:
                    messages.append({"role": role, "content": content})

        final_user_content = self._assemble_context(
            gender, age, query, chunks
        )
        messages.append({"role": "user", "content": final_user_content})
        return messages

    def answer(
        self,
        gender: str,
        age: int,
        query: str,
        system_prompt: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Retrieval:
        """Generate an answer for a medical query."""
        if self._provider != "huggingface":
            raise RuntimeError("RagEngine is configured for Hugging Face usage only.")
        return self._answer_with_hf(
            gender,
            age,
            query,
            system_prompt,
            progress_callback,
            conversation_history,
        )

    def _answer_with_hf(
        self,
        gender: str,
        age: int,
        query: str,
        system_prompt: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Retrieval:
        if progress_callback:
            progress_callback("Loading embedding model...", 0.05)
        idx, kb = self._ensure_hf_index(), self._ensure_knowledge_base()
        if progress_callback:
            progress_callback("Building search query...", 0.1)
        search_q = self._build_retrieval_query(
            gender, age, query, conversation_history
        )
        if progress_callback:
            progress_callback("Embedding query...", 0.2)
        q_emb = self._hf_embed([search_q], prompt_name="query")[0]
        if progress_callback:
            progress_callback(f"Searching index ({len(kb.documents)} docs)...", 0.4)
        dists, idxs = idx.search(q_emb.reshape(1, -1), self._top_k)
        if progress_callback:
            progress_callback("Processing results...", 0.5)
        chunks = (
            [
                RetrievedChunk(
                    content=kb.documents[i].page_content,
                    source=kb.documents[i].metadata.get("source", "unknown"),
                    score=float(s),
                )
                for s, i in zip(dists[0], idxs[0])
                if 0 <= i < len(kb.documents)
            ]
            if idxs.size > 0
            else []
        )
        if progress_callback:
            progress_callback(f"Retrieved {len(chunks)} relevant chunks", 0.6)
        gen_messages = self._build_generation_messages(
            system_prompt,
            gender,
            age,
            query,
            chunks,
            conversation_history,
        )
        if progress_callback:
            progress_callback("Generating response with LLM...", 0.7)
        return Retrieval(
            answer=self._generate_hf_response(gen_messages),
            supporting_chunks=chunks,
            usage=None,
        )

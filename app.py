from __future__ import annotations

from typing import Any, Optional
from html import escape
import re

import bleach
from markdown import Markdown

import torch

import streamlit as st

from src.config import (
    BOT_NAME,
    CACHE_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_DOCS_DIR,
    GENERATION_CONFIG,
    HF_CHAT_MODEL,
    HF_EMBEDDING_MODEL,
    SYSTEM_PROMPT,
    TOP_K_RESULTS,
)
from src.rag_engine import RagEngine


_MD_CONVERTER = Markdown(extensions=["extra", "sane_lists", "smarty"])
_ALLOWED_TAGS = {
    "p",
    "strong",
    "em",
    "ul",
    "ol",
    "li",
    "code",
    "pre",
    "blockquote",
    "br",
    "a",
}
_ALLOWED_ATTRIBUTES = {"a": ["href", "title", "rel", "target"]}


def _markdown_to_html(text: str) -> str:
    html = _MD_CONVERTER.convert(text)
    _MD_CONVERTER.reset()
    html = html.strip()
    html = re.sub(r"(\n\s*){3,}", "\n\n", html)
    html = re.sub(r"\s{2,}", " ", html)
    cleaned = bleach.clean(
        html,
        tags=_ALLOWED_TAGS,
        attributes=_ALLOWED_ATTRIBUTES,
        strip=True,
    )
    return cleaned


st.set_page_config(
    page_title=f"{BOT_NAME} | Medical Knowledge Assistant",
    page_icon="🩺",
    layout="wide",
)


def _theme_css() -> str:
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Poppins:wght@300;400;600&display=swap');
:root { color-scheme: dark; }
body {
    background: radial-gradient(circle at top, #1e1a4d 10%, #0b0a1f 55%, #04040a 100%);
    color: #fefefe;
    font-family: 'Poppins', 'Segoe UI', sans-serif;
}

section[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, rgba(14, 9, 34, 0.95), rgba(10, 12, 40, 0.98));
    border-right: 2px solid rgba(255, 221, 85, 0.25);
    color: #fefefe;
    padding: 2.2rem 1.6rem 3rem 1.6rem;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {
    color: #ffde59;
    font-family: 'Press Start 2P', cursive;
    letter-spacing: 1px;
}

div.block-container {
    padding-top: 2.5rem;
    max-width: 1080px;
}

.medibot-hero {
    background: radial-gradient(circle at 20% 20%, rgba(255, 222, 89, 0.95), rgba(255, 101, 134, 0.85), rgba(87, 136, 255, 0.9));
    color: #1d1a38;
    padding: 2.4rem;
    border-radius: 28px;
    margin-bottom: 2.4rem;
    box-shadow: 0 28px 65px rgba(22, 21, 64, 0.55);
    position: relative;
    overflow: hidden;
}

.medibot-hero::after {
    content: "";
    position: absolute;
    inset: -40px;
    background: url('https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/113.png') no-repeat right 5% center / 200px;
    opacity: 0.25;
    pointer-events: none;
}

.medibot-hero h1 {
    font-family: 'Press Start 2P', cursive;
    font-size: 2rem;
    margin-bottom: 1rem;
}

.medibot-hero p {
    font-size: 1.05rem;
    font-weight: 400;
    max-width: 70%;
}

.medibot-card {
    background: rgba(18, 20, 52, 0.92);
    border-radius: 22px;
    padding: 2rem 2.2rem;
    box-shadow: 0 30px 70px rgba(8, 10, 30, 0.55);
    border: 2px solid rgba(255, 222, 89, 0.15);
}

.medibot-card h3 {
    margin-top: 0;
    color: #ffde59;
    font-family: 'Press Start 2P', cursive;
    font-size: 0.9rem;
    letter-spacing: 1px;
}

input, textarea, select, .stNumberInput input, .stTextInput input {
    border-radius: 14px !important;
    background-color: rgba(12, 14, 40, 0.92) !important;
    color: #fefefe !important;
    border: 1px solid rgba(255, 222, 89, 0.35) !important;
}

.stButton > button {
    width: 100%;
    border-radius: 999px;
    padding: 0.85rem 1.2rem;
    background: linear-gradient(135deg, #ffde59, #ff656e);
    color: #1d1a38;
    border: none;
    font-weight: 700;
    font-family: 'Press Start 2P', cursive;
    letter-spacing: 1px;
    box-shadow: 0 16px 35px rgba(255, 101, 110, 0.45);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #ffe27d, #ff7a85);
    transform: translateY(-1px);
}

.stAlert {
    border-radius: 18px;
    border: 1px solid rgba(255, 222, 89, 0.25);
    background: rgba(12, 14, 40, 0.85);
}

.medibot-response {
    background: linear-gradient(135deg, rgba(32, 34, 78, 0.95), rgba(59, 82, 160, 0.9));
    border-radius: 24px;
    border: 2px solid rgba(255, 222, 89, 0.18);
    padding: 2rem 2.3rem;
    box-shadow: 0 24px 55px rgba(15, 18, 54, 0.6);
    color: #fefefe;
}

.medibot-response .medibot-answer {
    white-space: pre-wrap;
    word-break: break-word;
}

.medibot-progress {
    width: 100%;
    height: 8px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 999px;
    overflow: hidden;
    margin: 1.2rem 0 1rem;
}

.medibot-progress__bar {
    height: 100%;
    width: 0%;
    background: linear-gradient(90deg, #ffde59 0%, #ff7a85 50%, #5d8eff 100%);
    transition: width 180ms ease-out;
}

.chat-bubble {
    max-width: 85%;
    padding: 1rem 1.2rem;
    border-radius: 18px;
    box-shadow: 0 18px 45px rgba(9, 11, 36, 0.45);
    position: relative;
    line-height: 1.6;
    margin-bottom: 0.5rem;
}

.chat-bubble.assistant {
    background: linear-gradient(135deg, rgba(46, 54, 114, 0.95), rgba(88, 110, 206, 0.85));
    color: #fefefe;
    border-bottom-left-radius: 6px;
}

.chat-bubble.user {
    background: linear-gradient(135deg, #7cf5c7, #3ac8a1);
    color: #0b1124;
    border-bottom-right-radius: 6px;
    margin-left: auto;
}

.chat-meta {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    opacity: 0.8;
    margin-bottom: 0;
}

.chat-meta-row {
    display: flex;
    align-items: center;
    margin-bottom: 0.35rem;
}

.chat-meta-row.assistant,
.chat-meta-row.user {
    justify-content: space-between;
}

.chat-text {
    white-space: normal;
    word-break: break-word;
    font-size: 0.97rem;
    line-height: 1.5;
}

.chat-text p {
    margin: 0 0 0.6rem;
}

.chat-text p:last-child {
    margin-bottom: 0;
}

.chat-text ul,
.chat-text ol {
    margin: 0.4rem 0 0.6rem;
    padding-left: 1.4rem;
    list-style-position: outside;
}

.chat-text ul:last-child,
.chat-text ol:last-child {
    margin-bottom: 0;
}

.chat-text ul {
    list-style-type: disc;
}

.chat-text ol {
    list-style-type: decimal;
}

div[data-testid="stChatMessage"] {
    padding: 0 !important;
    background: transparent !important;
}

div[data-testid="stChatMessage"] > div[data-testid="stChatMessageContent"] {
    padding: 0 !important;
    background: transparent !important;
}

div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] {
    padding: 0 !important;
    background: transparent !important;
}

/* Hide avatars for cleaner appearance */
div[data-testid="stChatMessage"] div[data-testid="stChatMessageAvatarContainer"] {
    display: none !important;
}

.medibot-context {
    background: rgba(14, 16, 44, 0.95);
    border-radius: 18px;
    padding: 1.3rem 1.6rem;
    border: 1px solid rgba(255, 222, 89, 0.18);
    margin-bottom: 1.1rem;
    color: #fefefe;
}

.medibot-context h4 {
    margin-bottom: 0.5rem;
    color: #ffde59;
    font-family: 'Press Start 2P', cursive;
    font-size: 0.8rem;
}

.stExpander > div {
    border-radius: 18px;
    border: 1px solid rgba(255, 222, 89, 0.22);
}

.medibot-footer {
    text-align: center;
    font-size: 0.85rem;
    color: rgba(255, 255, 255, 0.72);
    padding: 1.2rem 0 1.6rem;
    margin-top: 3rem;
    border-top: 1px solid rgba(255, 222, 89, 0.18);
}

/* Subtle gear animation for status */
[data-testid="stStatus"] [data-testid="stStatusWidget"]::before {
    content: "";
    display: inline-block;
    width: 14px;
    height: 14px;
    margin-right: 8px;
    background: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="%23ffde59"><path d="M19.14,12.94a7.66,7.66,0,0,0,.05-.94,7.66,7.66,0,0,0-.05-.94l2.11-1.65a.48.48,0,0,0,.12-.6l-2-3.46a.5.5,0,0,0-.58-.22L16.7,5.88a7.85,7.85,0,0,0-1.63-.94l-.25-2.65A.49.49,0,0,0,14.33,2H9.67a.49.49,0,0,0-.49.43L9,5.08a7.85,7.85,0,0,0-1.63.94L4.21,4.13a.5.5,0,0,0-.58.22l-2,3.46a.48.48,0,0,0,.12.6L3.86,10.06a7.66,7.66,0,0,0-.05.94,7.66,7.66,0,0,0,.05.94L1.75,13.59a.48.48,0,0,0-.12.6l2,3.46a.5.5,0,0,0,.58.22L7.3,18.12a7.85,7.85,0,0,0,1.63.94l.25,2.65a.49.49,0,0,0,.49.43h4.66a.49.49,0,0,0,.49-.43l.25-2.65a7.85,7.85,0,0,0,1.63-.94l2.51,1.75a.5.5,0,0,0,.58-.22l2-3.46a.48.48,0,0,0-.12-.6ZM12,15.5A3.5,3.5,0,1,1,15.5,12,3.5,3.5,0,0,1,12,15.5Z"/></svg>') no-repeat center/contain;
    animation: spin 1s linear infinite;
}

@keyframes spin { to { transform: rotate(360deg); } }
</style>
"""


def _apply_theme() -> None:
    st.markdown(_theme_css(), unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_engine(
    api_key: Optional[str],
    docs_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
    chat_model: str,
    top_k: int,
    provider: str,
    generation_config: Optional[dict[str, Any]] = None,
) -> RagEngine:
    if provider != "huggingface":
        raise ValueError("load_engine supports only the 'huggingface' provider")
    generation_config = generation_config or GENERATION_CONFIG
    return RagEngine(
        provider=provider,
        api_key=api_key,
        docs_dir=docs_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        chat_model=chat_model,
        top_k=top_k,
        generation_config=generation_config,
    )


def _render_sidebar(
    provider: str,
    demo_mode: bool,
    provider_label: str,
    chat_model: str,
    embedding_model: str,
    total_time: float | None,
    *,
    container: Any | None = None,
) -> None:
    if container is not None:
        container.empty()
        panel = container.container()
    else:
        panel = st.sidebar
    with panel:
        st.markdown("## Control Panel")
        if demo_mode:
            st.warning(
                "Demo mode is active. Connect a CUDA-capable GPU to unlock live guidance.",
                icon="⚠️",
            )
        else:
            st.success("CUDA GPU detected. Ready to assist clinicians.", icon="✅")
        st.markdown("---")
        if not demo_mode:
            gen_summary = ", ".join(f"{k}={v}" for k, v in GENERATION_CONFIG.items())
            st.caption(
                f"Chat: `{chat_model}`  \n"
                f"Embeddings: `{embedding_model}`  \n"
                f"Docs Dir: `{DEFAULT_DOCS_DIR}`  \n"
                f"Cache Dir: `{CACHE_DIR}`  \n"
                f"Chunk Size: `{CHUNK_SIZE}`  \n"
                f"Chunk Overlap: `{CHUNK_OVERLAP}`  \n"
                f"Top K Results: `{TOP_K_RESULTS}`  \n"
                f"Generation: `{gen_summary}`"
            )
            if total_time is not None:
                st.caption(f"Last inference time: `{total_time:.2f}s`")
            st.markdown("---")


def _validate_inputs(gender: str, age: int, query: str) -> list[str]:
    issues = []
    if gender == "Select gender":
        issues.append("Please choose a gender option.")
    if age <= 0:
        issues.append("Please enter an age greater than 0.")
    if not query.strip():
        issues.append("Describe the medical question or symptoms.")
    return issues


def _render_chat(messages: list[dict[str, Any]]) -> None:
    if not messages:
        st.info("Start the consultation by sharing the user's concern.")
        return

    for msg in messages:
        role = msg.get("role", "assistant")
        role_key = "user" if role == "user" else "assistant"
        raw_content = msg.get("content", "") or ""
        html_text = _markdown_to_html(raw_content)
        chat_container = st.chat_message(role_key)

        if role_key == "user":
            meta_html = "<div class='chat-meta-row user'><span class='chat-meta'>You</span></div>"
        else:
            meta_html = (
                f"<div class='chat-meta-row assistant'>"
                f"<span class='chat-meta'>{BOT_NAME}</span>"
                "</div>"
            )

        with chat_container:
            st.markdown(
                f"<div class='chat-bubble {role_key}'>{meta_html}<div class='chat-text'>{html_text}</div></div>",
                unsafe_allow_html=True,
            )
            if role_key == "assistant" and msg.get("chunks"):
                with st.expander("View retrieved context"):
                    for idx, chunk in enumerate(msg.get("chunks", []), 1):
                        source = escape(str(chunk.get("source", "unknown")))
                        score = chunk.get("score")
                        score_str = (
                            f"{float(score):.4f}"
                            if isinstance(score, (int, float))
                            else "–"
                        )
                        st.markdown(f"**[{idx}] `{source}`** · Relevance {score_str}")
                        content = chunk.get("content", "") or ""
                        st.code(content.strip() or "(empty)", language="markdown")


def main() -> None:
    has_cuda = torch.cuda.is_available()
    provider = "huggingface" if has_cuda else "demo"
    demo_mode = not has_cuda
    api_key: Optional[str] = None

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": f"Hello, I'm {BOT_NAME}. Share your query and I'll reference the knowledge base for you.",
            }
        ]
    if "last_total_time" not in st.session_state:
        st.session_state.last_total_time = None

    embedding_model = HF_EMBEDDING_MODEL
    chat_model = HF_CHAT_MODEL if has_cuda else "N/A"
    provider_label = "Hugging Face · Qwen3 (GPU)" if has_cuda else "Demo"

    control_panel = st.sidebar.empty()
    _apply_theme()

    st.markdown(
        f"""
        <div class="medibot-hero">
            <h1>{BOT_NAME} Copilot</h1>
            <p>Keep the dialogue going—each follow-up draws on prior messages and the institutional knowledge base to stay precise.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if demo_mode:
        st.info(
            "Demo mode is active. Attach this application to a CUDA-capable GPU to generate tailored guidance."
        )

    st.markdown("### User Snapshot")
    gender_options = [
        "Select gender",
        "Female",
        "Male",
        "Non-binary",
        "Prefer not to say",
    ]
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox(
            "Gender", options=gender_options, index=0, label_visibility="visible"
        )
    with col2:
        age = st.number_input("Age", min_value=0, max_value=120, step=1, value=30)

    toolbar_left, toolbar_right = st.columns([3, 1])
    with toolbar_left:
        reset = st.button(
            "↺ Reset Conversation", use_container_width=True, type="secondary"
        )
    with toolbar_right:
        kill = st.button(
            "🛑 STOP",
            use_container_width=True,
            type="secondary",
            help="Halts generation and clears GPU memory/caches.",
        )

    if reset:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": f"Conversation cleared. I'm ready for new details whenever you are.",
            }
        ]
        st.session_state.last_total_time = None
        st.experimental_rerun()

    if kill:
        try:
            if provider == "huggingface" and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    st.success("✓ GPU memory cleared")
                except Exception:
                    pass
            st.cache_resource.clear()
            st.cache_data.clear()
            st.warning("🛑 Process killed. All caches cleared. GPU memory freed.")
            st.stop()
        except Exception as exc:
            st.error(f"Error during kill: {exc}")

    st.caption(f"Message {BOT_NAME}")
    user_prompt = st.chat_input(
        "Describe the case or ask a follow-up...",
        disabled=demo_mode,
        key="chat_input",
    )

    if user_prompt:
        cleaned_prompt = user_prompt.strip()
        if demo_mode:
            st.warning(
                "Demo mode is active. Attach a CUDA-capable GPU to generate responses."
            )
        elif errors := _validate_inputs(gender, int(age), cleaned_prompt):
            for issue in errors:
                st.error(issue)
        else:
            history = list(st.session_state.messages)
            st.session_state.messages.append({"role": "user", "content": cleaned_prompt})
            try:
                with st.spinner(f"Preparing {provider_label} pipeline..."):
                    engine = load_engine(
                        api_key=api_key,
                        docs_dir=str(DEFAULT_DOCS_DIR),
                        chunk_size=CHUNK_SIZE,
                        chunk_overlap=CHUNK_OVERLAP,
                        embedding_model=embedding_model,
                        chat_model=chat_model,
                        top_k=TOP_K_RESULTS,
                        provider=provider,
                        generation_config=GENERATION_CONFIG,
                    )

                progress_placeholder = st.empty()

                def _render_progress(fraction: float) -> None:
                    clamped = min(max(fraction, 0.0), 1.0) * 100
                    progress_placeholder.markdown(
                        f"""
                        <div class="medibot-progress">
                            <div class="medibot-progress__bar" style="width: {clamped:.2f}%;"></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                def update_progress(_: str, frac: float) -> None:
                    _render_progress(frac)

                import time

                _render_progress(0.0)
                t0 = time.time()
                result = engine.answer(
                    gender,
                    int(age),
                    cleaned_prompt,
                    SYSTEM_PROMPT,
                    update_progress,
                    conversation_history=history,
                )
                total_time = time.time() - t0
                _render_progress(1.0)
                progress_placeholder.empty()

                chunks_payload = [
                    {
                        "content": chunk.content,
                        "source": chunk.source,
                        "score": chunk.score,
                    }
                    for chunk in result.supporting_chunks
                ]
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result.answer,
                        "chunks": chunks_payload,
                    }
                )
                st.session_state.last_total_time = total_time
            except FileNotFoundError:
                st.error(
                    "Knowledge base directory not found. Ensure it exists and contains markdown files."
                )
            except RuntimeError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"The assistant could not complete the request: {exc}")

    st.markdown("### Conversation Thread")
    _render_chat(st.session_state.messages)

    _render_sidebar(
        provider=provider,
        demo_mode=demo_mode,
        provider_label=provider_label,
        chat_model=chat_model,
        embedding_model=embedding_model,
        total_time=st.session_state.get("last_total_time"),
        container=control_panel,
    )

    st.markdown(
        f"""
        <div class="medibot-footer">
            {BOT_NAME} provides informational guidance only. Consult licensed healthcare professionals for medical care.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

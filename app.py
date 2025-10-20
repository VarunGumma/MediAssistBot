from __future__ import annotations

from typing import Any, Optional
from html import escape

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
    background: url('https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/25.png') no-repeat right 5% center / 200px;
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


def main() -> None:
    has_cuda = torch.cuda.is_available()
    provider = "huggingface" if has_cuda else "demo"
    demo_mode = not has_cuda
    api_key: Optional[str] = None

    embedding_model = HF_EMBEDDING_MODEL
    chat_model = HF_CHAT_MODEL if has_cuda else "N/A"
    provider_label = "Hugging Face · Qwen3 (GPU)" if has_cuda else "Demo"

    control_panel = st.sidebar.empty()
    results_panel = st.sidebar.empty()

    _render_sidebar(
        provider=provider,
        demo_mode=demo_mode,
        provider_label=provider_label,
        chat_model=chat_model,
        embedding_model=embedding_model,
        total_time=None,
        container=control_panel,
    )
    _apply_theme()

    st.markdown(
        f"""
        <div class="medibot-hero">
            <h1>{BOT_NAME} Copilot</h1>
            <p>Share the patient's context and ask targeted questions. {BOT_NAME} returns evidence-grounded guidance sourced from your private knowledge base—wrapped in a playful, Pokémon-inspired interface.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if demo_mode:
        st.info(
            "Demo mode is active. Attach this application to a CUDA-capable GPU to generate tailored guidance."
        )

    with st.container():
        st.markdown("### Patient Snapshot")
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

    st.markdown("### Compose Message")
    if demo_mode:
        st.caption("GPU: not detected (demo mode).")

    query = st.text_area(
        "Describe the case or ask for guidance",
        height=220,
        placeholder="Example: Persistent cough for two weeks with mild fever. Possible causes and triage guidance? (Feel free to write in any language.)",
        help=f"{BOT_NAME} understands multilingual input for single-turn queries.",
        disabled=demo_mode,
    )

    col_submit, col_kill = st.columns([3, 1])
    with col_submit:
        submit = st.button(
            "Generate Guidance", use_container_width=True, disabled=demo_mode
        )
    with col_kill:
        kill = st.button(
            "🛑 Kill",
            use_container_width=True,
            type="secondary",
            help="Emergency stop: halts generation and clears GPU memory",
        )

    if kill:
        try:
            if provider == "huggingface" and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    st.success("✓ GPU memory cleared")
                except:
                    pass
            st.cache_resource.clear()
            st.cache_data.clear()
            st.warning("🛑 Process killed. All caches cleared. GPU memory freed.")
            st.stop()
        except Exception as exc:
            st.error(f"Error during kill: {exc}")

    if submit:
        if demo_mode:
            st.warning(
                "Demo mode is active. Attach a CUDA-capable GPU to generate responses."
            )
        elif errors := _validate_inputs(gender, int(age), query):
            for issue in errors:
                st.error(issue)
        else:
            results_panel.empty()
            try:
                total_time = None
                with st.spinner(f"Initializing {provider_label} pipeline..."):
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

                with st.status(
                    "🔍 Retrieving Context ...", expanded=True
                ) as status_retrieve:
                    status_retrieve.write(
                        f"Searching knowledge base for relevant documents and generating response... "
                    )
                    progress_bar, progress_text = st.progress(0.0), st.empty()

                    def update_progress(msg: str, frac: float):
                        progress_bar.progress(frac)
                        progress_text.text(msg)

                    import time

                    t0 = time.time()
                    result = engine.answer(
                        gender, int(age), query.strip(), SYSTEM_PROMPT, update_progress
                    )
                    total_time = time.time() - t0
                    progress_bar.progress(1.0)
                    progress_text.empty()
                    n_chunks = len(result.supporting_chunks or [])
                    status_retrieve.write(f"✓ Retrieved {n_chunks} relevant chunks")
                    status_retrieve.update(
                        label=f"✅ Retrieved {n_chunks} chunks", state="complete"
                    )

            except FileNotFoundError:
                st.error(
                    "Knowledge base directory not found. Ensure it exists and contains markdown files."
                )
            except RuntimeError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"The assistant could not complete the request: {exc}")
            else:
                # Display the answer
                st.markdown(
                    f"""
                    <div class="medibot-response">
                        <h3>🩺 Guidance from {BOT_NAME}</h3>
                        <div class="medibot-answer" style="line-height:1.7;">{escape(result.answer)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                results_panel.empty()
                with results_panel.container():
                    if result.usage and result.usage.get("total_tokens"):
                        st.caption(
                            f"Token usage (total): `{result.usage['total_tokens']}`"
                        )
                    if result.supporting_chunks:
                        st.markdown("### 📚 Retrieved Context")
                        st.caption(
                            f"Found {len(result.supporting_chunks)} supporting chunk(s). Expand to inspect sources."
                        )
                        with st.expander(
                            f"📖 View {len(result.supporting_chunks)} Source Documents",
                            expanded=False,
                        ):
                            for idx, chunk in enumerate(result.supporting_chunks, 1):
                                st.markdown(f"**[{idx}] {chunk.source}**")
                                st.markdown(f"*Relevance Score: {chunk.score:.4f}*")
                                st.markdown(
                                    f'<div style="background: rgba(255, 222, 89, 0.08); padding: 0.75rem; border-radius: 8px; margin-bottom: 0.75rem; border-left: 3px solid rgba(255, 222, 89, 0.5); max-height: 200px; overflow:auto; white-space: pre-wrap;">{escape(chunk.content.strip())}</div>',
                                    unsafe_allow_html=True,
                                )
                        st.markdown("#### 📋 Quick Source List")
                        for src in sorted({c.source for c in result.supporting_chunks}):
                            st.markdown(f"- `{src}`")
                    else:
                        st.caption(
                            "No supporting context retrieved for this query. The response may be less reliable."
                        )

                _render_sidebar(
                    provider=provider,
                    demo_mode=demo_mode,
                    provider_label=provider_label,
                    chat_model=chat_model,
                    embedding_model=embedding_model,
                    total_time=total_time,
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

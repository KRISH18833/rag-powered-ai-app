import os
import time
from pathlib import Path
import streamlit as st
from src.config import get_gemini_api_key, DB_DIR, COLLECTION_NAME
from src.vectordb.vector_store import get_chroma_client
from src.retrieval.retriever import retrieve
from src.llm.llm_client import (
    rag_answer,
    plain_answer,
    GenerationFailure,
    classify_gemini_error,
    format_degraded_passages,
)
from src.utils.ui import apply_video_background

st.set_page_config(page_title="Cloud Notes RAG", page_icon="☁️", layout="wide")

# ── Video background & Glassmorphism ──────────────────────────────────────────
apply_video_background()

# ── Auto-ingest index if missing at startup ───────────────────────────────────
@st.cache_resource(show_spinner=False)
def _run_guarded_ingest():
    """Run ingestion guarded across sessions by st.cache_resource."""
    try:
        client = get_chroma_client()
        col = client.get_collection(COLLECTION_NAME)
        if col.count() > 0:
            return True
    except Exception:
        pass

    from ingest import main as run_ingest
    run_ingest()
    return True

def ensure_vector_db():
    """Ensure vector database exists and is populated."""
    try:
        client = get_chroma_client()
        col = client.get_collection(COLLECTION_NAME)
        if col.count() > 0:
            return
    except Exception:
        pass

    with st.spinner("Building vector index from course notes..."):
        _run_guarded_ingest()

ensure_vector_db()

# ── API Key Setup Check ───────────────────────────────────────────────────────
api_key = get_gemini_api_key()
if not api_key:
    st.error("⚠️ GEMINI_API_KEY is missing. Please set it in your environment variables, Streamlit secrets, or .env file.")
    st.stop()

# ── Cached answer functions ───────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def cached_rag_answer(question):
    return rag_answer(question)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_plain_answer(question):
    return plain_answer(question)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_retrieve(question):
    return retrieve(question)

# ── Session state ──────────────────────────────────────────────────────────────
if "question" not in st.session_state:
    st.session_state.question = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "auto_submit" not in st.session_state:
    st.session_state.auto_submit = False

SAMPLE_QUESTIONS = [
    (
        "W&B Model Version 2",
        "In the W&B versioning example, which command loads version 2 of the model, and what file name is used?",
    ),
    (
        "STREAM Benchmark (75 EC2)",
        "What bandwidth did the STREAM benchmark show with 75 EC2 instances?",
    ),
    (
        "Hadoop Job Components",
        "In Hadoop, what three components contribute to running a job?",
    ),
]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    compare_mode = st.toggle("Compare with plain Gemini (no retrieval)", value=False)
    st.markdown("---")
    st.markdown("""
    **☁️ Course Assistant**  
    RAG system grounded strictly on *Cloud Computing for Machine Learning* course notes.
    """)
    st.markdown("---")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.question = ""
        st.session_state.auto_submit = False
        st.rerun()

# ── Header card ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="glass" style="text-align:center; padding: 2rem 2rem 1.4rem;">
  <h1 style="font-size:2.4rem; margin:0 0 0.3rem;">☁️ Cloud Notes RAG</h1>
  <p style="color:rgba(255,255,255,0.65) !important; font-size:1.05rem; margin:0;">
    Retrieval-Augmented Generation over Cloud Computing for ML course notes &nbsp;·&nbsp; Chroma + Gemini
  </p>
</div>
""", unsafe_allow_html=True)

# ── Sample question buttons ───────────────────────────────────────────────────
st.markdown("<p style='color:rgba(255,255,255,0.55);font-size:0.85rem;margin-bottom:0.4rem;'>Try a course question:</p>", unsafe_allow_html=True)
ecols = st.columns(3)
for col, (label, q_text) in zip(ecols, SAMPLE_QUESTIONS):
    with col:
        if st.button(label, use_container_width=True):
            st.session_state.question = q_text
            st.session_state.auto_submit = True
            st.rerun()

# ── Input row ─────────────────────────────────────────────────────────────────
icol, bcol = st.columns([5, 1])
with icol:
    question = st.text_input(
        "question_input",
        value=st.session_state.question,
        placeholder="Ask anything about your Cloud Computing notes…",
        label_visibility="collapsed",
    )
with bcol:
    ask = st.button("Ask ✦", use_container_width=True)

st.markdown("<hr style='border-color:rgba(255,255,255,0.1);margin:0.5rem 0 1rem;'>", unsafe_allow_html=True)

# ── Helper: Status badge ───────────────────────────────────────────────────────
def render_status_badge(status):
    if status == "Retrieved from course notes":
        st.markdown("""
        <div style="margin-bottom: 0.5rem;">
          <span style="background: rgba(52, 152, 219, 0.2); border: 1px solid rgba(52, 152, 219, 0.5); color: #3498db !important; border-radius: 12px; padding: 3px 12px; font-size: 0.82rem; font-weight: 600; display: inline-flex; align-items: center; gap: 5px;">
            ● Retrieved from course notes
          </span>
        </div>
        """, unsafe_allow_html=True)
    elif status == "Answered from course notes" or status is True:
        st.markdown("""
        <div style="margin-bottom: 0.5rem;">
          <span style="background: rgba(46, 204, 113, 0.2); border: 1px solid rgba(46, 204, 113, 0.5); color: #2ecc71 !important; border-radius: 12px; padding: 3px 12px; font-size: 0.82rem; font-weight: 600; display: inline-flex; align-items: center; gap: 5px;">
            ● Answered from course notes
          </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="margin-bottom: 0.5rem;">
          <span style="background: rgba(231, 76, 60, 0.2); border: 1px solid rgba(231, 76, 60, 0.5); color: #e74c3c !important; border-radius: 12px; padding: 3px 12px; font-size: 0.82rem; font-weight: 600; display: inline-flex; align-items: center; gap: 5px;">
            ○ Not in course notes
          </span>
        </div>
        """, unsafe_allow_html=True)

# ── Helper: Source chips ───────────────────────────────────────────────────────
def render_chips(sources):
    if sources:
        chips = "".join(f'<span class="chip">{s}</span>' for s in sources)
        st.markdown(f"<div style='margin-top:0.5rem; margin-bottom: 0.5rem;'>{chips}</div>", unsafe_allow_html=True)

# ── Helper: Sources panel ─────────────────────────────────────────────────────
def render_sources_panel(chunks, is_in_notes: bool):
    with st.expander("Sources", expanded=False):
        if not chunks or not is_in_notes:
            st.markdown("<p style='color: rgba(255,255,255,0.7); font-size: 0.85rem;'>No matching sources found within course note relevance threshold.</p>", unsafe_allow_html=True)
        else:
            for i, c in enumerate(chunks, 1):
                text_preview = c["text"][:300].replace("\n", " ").strip()
                dist = c.get("distance", 0.0)
                st.markdown(f"""
                <div class="chunk-card">
                  <div class="chunk-meta">#{i} &nbsp;|&nbsp; <b>{c['source']}</b> &nbsp;·&nbsp; page {c['page']} &nbsp;·&nbsp; distance <code>{dist:.4f}</code></div>
                  <div style="color: #ddd;">{text_preview}{'…' if len(c['text']) > 300 else ''}</div>
                </div>
                """, unsafe_allow_html=True)

# ── Process new question ───────────────────────────────────────────────────────
should_run = (ask or st.session_state.auto_submit) and question.strip()

if should_run:
    st.session_state.auto_submit = False
    st.session_state.question = question

    with st.spinner("Searching course notes..."):
        try:
            res = cached_rag_answer(question)
            rag_res = res[0]
            sources = res[1]
            chunks = getattr(res, "chunks", [])
            is_in_notes = (rag_res.strip() != "I couldn't find this in the course notes." and len(sources) > 0)
            badge_status = "Answered from course notes" if is_in_notes else "Not in course notes"

            plain_res = None
            if compare_mode:
                try:
                    plain_res = cached_plain_answer(question)
                except Exception as pe:
                    exc_type = type(pe).__name__
                    exc_msg = str(pe)
                    print(f"[STREAMLIT LOGS] plain_answer error - {exc_type}: {exc_msg}", flush=True)
                    plain_res = "AI summary is temporarily unavailable for direct comparison."

            st.session_state.chat_history.append({
                "question": question,
                "answer": rag_res,
                "sources": sources,
                "chunks": chunks,
                "is_in_notes": is_in_notes,
                "badge_status": badge_status,
                "is_degraded": False,
                "plain_answer": plain_res,
                "compare_mode": compare_mode,
            })
        except GenerationFailure as gf:
            # Graceful degradation: retrieval succeeded, generation failed after all retries
            exc = gf.original_exc
            exc_type = type(exc).__name__
            exc_msg = str(exc)
            print(f"[STREAMLIT LOGS] GenerationFailure after retries - {exc_type}: {exc_msg}", flush=True)
            passages_text = format_degraded_passages(gf.chunks)
            st.session_state.chat_history.append({
                "question": question,
                "answer": passages_text,
                "sources": gf.sources,
                "chunks": gf.chunks,
                "is_in_notes": True,
                "badge_status": "Retrieved from course notes",
                "is_degraded": True,
                "plain_answer": None,
                "compare_mode": False,
            })
        except Exception as e:
            friendly_err = classify_gemini_error(e)
            st.error(friendly_err)
            if st.button("🔄 Retry", key="retry_error_btn"):
                st.session_state.auto_submit = True
                st.rerun()

elif (ask or st.session_state.auto_submit) and not question.strip():
    st.session_state.auto_submit = False
    st.markdown('<div class="err-box">Please enter a question first.</div>', unsafe_allow_html=True)

# ── Render Chat History ────────────────────────────────────────────────────────
if st.session_state.chat_history:
    for idx, item in enumerate(st.session_state.chat_history):
        st.markdown(f"<p style='color:rgba(200,190,255,0.9); font-weight:600; font-size:1.05rem; margin-top:1.2rem; margin-bottom:0.4rem;'>🧑‍💻 {item['question']}</p>", unsafe_allow_html=True)

        badge = item.get("badge_status", "Answered from course notes" if item["is_in_notes"] else "Not in course notes")

        if item.get("compare_mode") and item.get("plain_answer"):
            lcol, rcol = st.columns(2)
            with lcol:
                st.markdown(f"""
                <div class="glass-sm">
                  <h4 style="margin:0 0 0.8rem; color:rgba(255,180,100,1) !important;">🌐 Without RAG</h4>
                  {item['plain_answer']}
                </div>
                """, unsafe_allow_html=True)

            with rcol:
                render_status_badge(badge)
                st.markdown(f'<div class="glass-sm">{item["answer"]}</div>', unsafe_allow_html=True)
                render_chips(item["sources"])
                render_sources_panel(item["chunks"], item["is_in_notes"])
        else:
            render_status_badge(badge)
            st.markdown(f'<div class="glass">{item["answer"]}</div>', unsafe_allow_html=True)
            render_chips(item["sources"])
            render_sources_panel(item["chunks"], item["is_in_notes"])

        if item.get("is_degraded"):
            if st.button("🔄 Retry", key=f"retry_degraded_{idx}"):
                st.session_state.question = item["question"]
                st.session_state.auto_submit = True
                st.rerun()

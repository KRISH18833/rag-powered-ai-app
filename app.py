import os
import time
import uuid
from pathlib import Path
import streamlit as st
from src.config import (
    get_gemini_api_key,
    DB_DIR,
    COLLECTION_NAME,
    UPLOAD_MAX_FILE_MB,
)
from src.vectordb.vector_store import (
    get_chroma_client,
    index_uploaded,
    list_uploaded,
    remove_uploaded,
    clear_uploaded,
)
from src.ingestion.loader import load_pdf_bytes, UploadError
from src.chunking.chunker import chunk_pages
from src.retrieval.retriever import retrieve
from src.prompts.prompt_templates import FALLBACK_RESPONSE
from src.llm.llm_client import (
    rag_answer,
    plain_answer,
    GenerationFailure,
    classify_gemini_error,
    format_degraded_passages,
    get_upload_hash,
)
from src.utils.ui import apply_video_background

st.set_page_config(page_title="Cloud Notes RAG", page_icon="☁️", layout="wide")

# ── Video background & Glassmorphism ──────────────────────────────────────────
apply_video_background()

# ── Auto-ingest course notes if missing at startup ─────────────────────────────
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
    """Ensure course notes vector database exists and is populated."""
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
def cached_rag_answer(question, scope="course", session_id=None, upload_hash=None):
    return rag_answer(question, scope=scope, session_id=session_id, upload_hash=upload_hash)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_plain_answer(question):
    return plain_answer(question)


# ── Session state ──────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "question" not in st.session_state:
    st.session_state.question = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "auto_submit" not in st.session_state:
    st.session_state.auto_submit = False

# Scope state — managed via st.session_state.scope + scope_override flag pattern
if "scope" not in st.session_state:
    st.session_state.scope = "Course notes (Cloud Computing for ML)"

if "scope_user_set" not in st.session_state:
    st.session_state.scope_user_set = False

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# Process scope_override BEFORE the radio is created so the new value is
# already in st.session_state.scope when the radio renders.
if "scope_override" in st.session_state:
    st.session_state.scope = st.session_state.pop("scope_override")

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

    st.markdown("### 📄 Chat with your own PDFs")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Up to 5 PDFs, 10 MB each. Larger PDFs take longer to index. Your files stay in this session only and are never saved.",
        label_visibility="collapsed",
    )
    st.caption("Up to 5 PDFs, 10 MB each. Larger PDFs take longer to index. Your files stay in this session only and are never saved.")

    if st.button("📥 Index my PDFs", use_container_width=True):
        if not uploaded_files:
            st.warning("Please choose at least one PDF file first.")
        else:
            progress_placeholder = st.empty()
            newly_indexed_any = False

            for f in uploaded_files:
                filename = f.name
                file_bytes = f.getvalue()

                # Fast check for file size
                if len(file_bytes) > UPLOAD_MAX_FILE_MB * 1024 * 1024:
                    st.error(f"'{filename}' exceeds the maximum allowed size of {UPLOAD_MAX_FILE_MB:.0f} MB.")
                    continue

                progress_bar = progress_placeholder.progress(0, text=f"Reading {filename}...")
                try:
                    pages = load_pdf_bytes(file_bytes, filename)
                    chunks = chunk_pages(pages)

                    def progress_cb(current, total, msg=None):
                        pct = int((current / max(total, 1)) * 100)
                        status_text = msg if msg else f"Embedding {filename}: {current}/{total} chunks..."
                        progress_bar.progress(pct, text=status_text)

                    res = index_uploaded(
                        session_id=st.session_state.session_id,
                        filename=filename,
                        chunks=chunks,
                        file_bytes=file_bytes,
                        progress_callback=progress_cb,
                    )

                    if res == "duplicate":
                        st.warning(f"'{filename}' was already indexed.")
                    else:
                        st.success(f"Indexed **{filename}** ({len(pages)} pages, {res} chunks)")
                        newly_indexed_any = True
                        st.session_state.processed_files.add(filename)

                except UploadError as ue:
                    st.error(f"'{filename}': {ue}")
                except Exception as e:
                    print(f"[STREAMLIT LOGS] Error indexing {filename}: {type(e).__name__}: {e}", flush=True)
                    st.error(f"Failed to index '{filename}': {type(e).__name__}")
                finally:
                    progress_placeholder.empty()

            if newly_indexed_any and not st.session_state.scope_user_set:
                st.session_state.scope_override = "My uploaded PDFs"
                st.toast("🗂️ Now chatting with your uploaded PDFs!", icon="✅")
                st.rerun()

    # List indexed documents in session
    indexed_docs = list_uploaded(st.session_state.session_id)
    if indexed_docs:
        st.markdown("#### Indexed Documents")
        for doc in indexed_docs:
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(
                    f"<div class='doc-card'><b>{doc['filename']}</b><br><span style='font-size:0.75rem; color:#d8b4fe;'>{doc['page_count']} pages · {doc['chunk_count']} chunks</span></div>",
                    unsafe_allow_html=True,
                )
            with c2:
                if st.button("🗑️", key=f"del_{doc['filename']}", help=f"Remove {doc['filename']}"):
                    remove_uploaded(st.session_state.session_id, doc['filename'])
                    st.session_state.processed_files.discard(doc['filename'])
                    # If no uploads remain, revert scope to course notes
                    remaining = list_uploaded(st.session_state.session_id)
                    if not remaining:
                        st.session_state.scope_override = "Course notes (Cloud Computing for ML)"
                        st.session_state.scope_user_set = False
                    st.rerun()

        if st.button("🗑️ Remove all my PDFs", use_container_width=True):
            clear_uploaded(st.session_state.session_id)
            st.session_state.processed_files.clear()
            st.session_state.scope_override = "Course notes (Cloud Computing for ML)"
            st.session_state.scope_user_set = False
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <p style='font-size:0.8rem; color:rgba(255,255,255,0.6); margin:0;'>
    ℹ️ <i>Uploaded documents are cleared when you close or refresh the page.</i>
    </p>
    """, unsafe_allow_html=True)
    st.markdown("---")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.question = ""
        st.session_state.auto_submit = False
        st.rerun()

# ── Header card ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="glass" style="text-align:center; padding: 1.8rem 2rem 1.4rem;">
  <h1 style="font-size:2.3rem; margin:0 0 0.3rem;">☁️ Cloud Notes & Document RAG</h1>
  <p style="color:rgba(255,255,255,0.7) !important; font-size:1.02rem; margin:0 0 0.6rem;">
    Grounded retrieval over Cloud Computing course notes & your private uploaded PDFs
  </p>
  <div style="font-size:0.85rem; color:rgba(255,255,255,0.6); display:flex; justify-content:center; gap:1.5rem; flex-wrap:wrap;">
    <span>📚 <b>Course Notes</b>: Cloud ML Units 1–5</span>
    <span>🔒 <b>Your Uploads</b>: In-memory & isolated to your session</span>
    <span>⚖️ <b>Both</b>: Unified cross-document retrieval</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Scope Selector ─────────────────────────────────────────────────────────────
SCOPE_CHOICES = [
    "My uploaded PDFs",
    "Course notes (Cloud Computing for ML)",
    "Both",
]

# Guard: scope must be a valid option
if st.session_state.scope not in SCOPE_CHOICES:
    st.session_state.scope = "Course notes (Cloud Computing for ML)"


def _on_scope_change():
    """Mark that the user has explicitly chosen a scope."""
    st.session_state.scope_user_set = True


st.radio(
    "Search Scope",
    SCOPE_CHOICES,
    key="scope",
    on_change=_on_scope_change,
    horizontal=True,
    label_visibility="collapsed",
)

scope_map = {
    "Course notes (Cloud Computing for ML)": "course",
    "My uploaded PDFs": "upload",
    "Both": "both",
}
current_scope = scope_map[st.session_state.scope]

# Check if user selected upload mode with no uploads
indexed_docs = list_uploaded(st.session_state.session_id)
has_uploads = len(indexed_docs) > 0
is_upload_only_and_empty = (current_scope == "upload" and not has_uploads)

if is_upload_only_and_empty:
    st.info("Upload a PDF in the sidebar to get started.")

# ── Active scope banner (shown above chat input) ──────────────────────────────
if current_scope == "upload":
    names = ", ".join(d["filename"] for d in indexed_docs) or "no files indexed"
    st.info(f"🔍 Searching: **your uploaded PDFs** ({names})")
elif current_scope == "both":
    names = ", ".join(d["filename"] for d in indexed_docs) or "none"
    st.info(f"🔍 Searching: **both** — your uploads ({names}) + Course notes")
else:
    st.info("🔍 Searching: **Course notes** (Cloud Computing for ML)")

# ── Sample question buttons / Scope hints ─────────────────────────────────────
if current_scope in ("course", "both"):
    st.markdown("<p style='color:rgba(255,255,255,0.55);font-size:0.85rem;margin-bottom:0.4rem;'>Try a course question:</p>", unsafe_allow_html=True)
    ecols = st.columns(3)
    for col, (label, q_text) in zip(ecols, SAMPLE_QUESTIONS):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state.question = q_text
                st.session_state.auto_submit = True
                st.rerun()
else:
    st.markdown("<p style='color:rgba(255,255,255,0.65);font-size:0.85rem;margin-bottom:0.4rem;'>💡 Ask something that only your document can answer</p>", unsafe_allow_html=True)

# ── Input row ─────────────────────────────────────────────────────────────────
icol, bcol = st.columns([5, 1])
with icol:
    question = st.text_input(
        "question_input",
        value=st.session_state.question,
        placeholder="Ask anything about your selected documents…" if not is_upload_only_and_empty else "Upload a PDF in the sidebar first…",
        label_visibility="collapsed",
        disabled=is_upload_only_and_empty,
    )
with bcol:
    ask = st.button("Ask ✦", use_container_width=True, disabled=is_upload_only_and_empty)

st.markdown("<hr style='border-color:rgba(255,255,255,0.1);margin:0.5rem 0 1rem;'>", unsafe_allow_html=True)

# ── Helper: Status badge ───────────────────────────────────────────────────────
def render_status_badge(status):
    if status == "Answered from your PDF":
        st.markdown("""
        <div style="margin-bottom: 0.5rem;">
          <span style="background: rgba(52, 152, 219, 0.2); border: 1px solid rgba(52, 152, 219, 0.5); color: #38bdf8 !important; border-radius: 12px; padding: 3px 12px; font-size: 0.82rem; font-weight: 600; display: inline-flex; align-items: center; gap: 5px;">
            ● Answered from your PDF
          </span>
        </div>
        """, unsafe_allow_html=True)
    elif status == "Answered from course notes":
        st.markdown("""
        <div style="margin-bottom: 0.5rem;">
          <span style="background: rgba(46, 204, 113, 0.2); border: 1px solid rgba(46, 204, 113, 0.5); color: #2ecc71 !important; border-radius: 12px; padding: 3px 12px; font-size: 0.82rem; font-weight: 600; display: inline-flex; align-items: center; gap: 5px;">
            ● Answered from course notes
          </span>
        </div>
        """, unsafe_allow_html=True)
    elif status == "Retrieved from your PDF":
        st.markdown("""
        <div style="margin-bottom: 0.5rem;">
          <span style="background: rgba(52, 152, 219, 0.2); border: 1px solid rgba(52, 152, 219, 0.5); color: #38bdf8 !important; border-radius: 12px; padding: 3px 12px; font-size: 0.82rem; font-weight: 600; display: inline-flex; align-items: center; gap: 5px;">
            ● Retrieved from your PDF
          </span>
        </div>
        """, unsafe_allow_html=True)
    elif status == "Retrieved from course notes":
        st.markdown("""
        <div style="margin-bottom: 0.5rem;">
          <span style="background: rgba(46, 204, 113, 0.2); border: 1px solid rgba(46, 204, 113, 0.5); color: #2ecc71 !important; border-radius: 12px; padding: 3px 12px; font-size: 0.82rem; font-weight: 600; display: inline-flex; align-items: center; gap: 5px;">
            ● Retrieved from course notes
          </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="margin-bottom: 0.5rem;">
          <span style="background: rgba(231, 76, 60, 0.2); border: 1px solid rgba(231, 76, 60, 0.5); color: #f87171 !important; border-radius: 12px; padding: 3px 12px; font-size: 0.82rem; font-weight: 600; display: inline-flex; align-items: center; gap: 5px;">
            ○ Not found in selected documents
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
            st.markdown("<p style='color: rgba(255,255,255,0.7); font-size: 0.85rem;'>No matching sources found within relevance threshold.</p>", unsafe_allow_html=True)
        else:
            for i, c in enumerate(chunks, 1):
                text_preview = c["text"][:300].replace("\n", " ").strip()
                dist = c.get("distance", 0.0)
                stype = c.get("source_type", "course")
                tag_class = "source-tag-upload" if stype == "upload" else "source-tag-course"
                tag_label = "Your upload" if stype == "upload" else "Course notes"
                fname = c.get("file", c.get("source", "Unknown"))
                st.markdown(f"""
                <div class="chunk-card">
                  <div class="chunk-meta">#{i} &nbsp;|&nbsp; <span class="source-tag {tag_class}">{tag_label}</span> &nbsp;|&nbsp; <b>{fname}</b> &nbsp;·&nbsp; page {c['page']} &nbsp;·&nbsp; distance <code>{dist:.4f}</code></div>
                  <div style="color: #ddd;">{text_preview}{'…' if len(c['text']) > 300 else ''}</div>
                </div>
                """, unsafe_allow_html=True)


# ── Process new question ───────────────────────────────────────────────────────
should_run = (ask or st.session_state.auto_submit) and question.strip() and not is_upload_only_and_empty

if should_run:
    st.session_state.auto_submit = False
    st.session_state.question = question

    spinner_text = "Searching uploaded documents..." if current_scope == "upload" else (
        "Searching course notes & uploads..." if current_scope == "both" else "Searching course notes..."
    )

    with st.spinner(spinner_text):
        try:
            upload_hash = get_upload_hash(st.session_state.session_id)
            res = cached_rag_answer(
                question,
                scope=current_scope,
                session_id=st.session_state.session_id,
                upload_hash=upload_hash,
            )
            rag_res = res[0]
            sources = res[1]
            chunks = getattr(res, "chunks", [])
            is_found = (rag_res.strip() != FALLBACK_RESPONSE and "couldn't find this" not in rag_res.lower() and len(sources) > 0)

            # Upload-only fallback tip: nudge user to also search course notes
            if current_scope == "upload" and not is_found:
                rag_res = rag_res + "\n\n*💡 Tip: switch to **Course notes** or **Both** to also search the built-in course material.*"

            if not is_found:
                badge_status = "Not found in selected documents"
            else:
                has_upload_chunk = any(c.get("source_type") == "upload" for c in chunks)
                has_course_chunk = any(c.get("source_type") == "course" for c in chunks)
                if has_upload_chunk and not has_course_chunk:
                    badge_status = "Answered from your PDF"
                elif has_course_chunk and not has_upload_chunk:
                    badge_status = "Answered from course notes"
                else:
                    top_stype = chunks[0].get("source_type", "course") if chunks else "course"
                    badge_status = "Answered from your PDF" if top_stype == "upload" else "Answered from course notes"

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
                "is_in_notes": is_found,
                "badge_status": badge_status,
                "is_degraded": False,
                "plain_answer": plain_res,
                "compare_mode": compare_mode,
                "scope": current_scope,
            })
        except GenerationFailure as gf:
            exc = gf.original_exc
            exc_type = type(exc).__name__
            exc_msg = str(exc)
            print(f"[STREAMLIT LOGS] GenerationFailure after retries - {exc_type}: {exc_msg}", flush=True)
            passages_text = format_degraded_passages(gf.chunks)
            has_upload_chunk = any(c.get("source_type") == "upload" for c in gf.chunks)
            degraded_badge = "Retrieved from your PDF" if has_upload_chunk else "Retrieved from course notes"
            st.session_state.chat_history.append({
                "question": question,
                "answer": passages_text,
                "sources": gf.sources,
                "chunks": gf.chunks,
                "is_in_notes": True,
                "badge_status": degraded_badge,
                "is_degraded": True,
                "plain_answer": None,
                "compare_mode": False,
                "scope": current_scope,
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
_scope_label_map = {
    "upload": "Your uploaded PDFs",
    "course": "Course notes",
    "both": "Both — uploads + Course notes",
}

if st.session_state.chat_history:
    for idx, item in enumerate(st.session_state.chat_history):
        st.markdown(f"<p style='color:rgba(200,190,255,0.9); font-weight:600; font-size:1.05rem; margin-top:1.2rem; margin-bottom:0.4rem;'>🧑‍💻 {item['question']}</p>", unsafe_allow_html=True)

        badge = item.get("badge_status", "Answered from course notes" if item["is_in_notes"] else "Not found in selected documents")
        item_scope = item.get("scope", "course")
        searched_label = _scope_label_map.get(item_scope, "Course notes")

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
                st.caption(f"Searched: {searched_label}")
                st.markdown(f'<div class="glass-sm">{item["answer"]}</div>', unsafe_allow_html=True)
                render_chips(item["sources"])
                render_sources_panel(item["chunks"], item["is_in_notes"])
        else:
            render_status_badge(badge)
            st.caption(f"Searched: {searched_label}")
            st.markdown(f'<div class="glass">{item["answer"]}</div>', unsafe_allow_html=True)
            render_chips(item["sources"])
            render_sources_panel(item["chunks"], item["is_in_notes"])

        if item.get("is_degraded"):
            if st.button("🔄 Retry", key=f"retry_degraded_{idx}"):
                st.session_state.question = item["question"]
                st.session_state.auto_submit = True
                st.rerun()

# Cloud Notes & Document RAG Assistant

A production-ready Retrieval-Augmented Generation (RAG) assistant built over **Cloud Computing for Machine Learning** course notes and **user-uploaded private PDFs**, powered by Google Gemini, ChromaDB, and Streamlit. The application features dual-scope retrieval, distance thresholding to eliminate hallucinations, resilient API failover with timeout guards, prompt-injection defense, and a glassmorphic web interface with a looping background video.

**Live demo:** [https://rag-powered-ai-app.streamlit.app/](https://rag-powered-ai-app.streamlit.app/)

---

## ✨ Features

- **Chat with Your Own PDFs**: Upload up to 5 PDFs (10 MB each, up to 150 pages and 400 chunks per session) and ask grounded questions directly over your own materials.
- **Three-Way Scope Selector**: Query across **"My uploaded PDFs"**, **"Course notes (Cloud Computing for ML)"**, or **"Both"** with unified retrieval and source attribution.
- **Strict Session Privacy**: Uploaded documents are processed entirely in-memory using `chromadb.EphemeralClient` and isolated to a unique visitor session ID. Files are never stored on disk, never shared across sessions, and are immediately purged when removed or when the browser session ends.
- **Memory & Quota Safeguards**: Enforces strict session limits (file size, total pages, total chunks) with clear remaining-capacity notifications. Includes a process-wide LRU eviction safeguard capped at 20 concurrent session collections.
- **Prompt-Injection Defense**: Grounded document-neutral system instructions treat retrieved text strictly as untrusted data, preventing adversarial prompt injections from overriding assistant persona or rules.
- **Resilient Model Fallback & Timeouts**: 15-second per-call timeout with automated rotation through reliable Gemini models (`gemini-3-flash-preview` → `gemini-3.5-flash` → `gemini-3.8-flash`). Transient errors (429/503) trigger exponential backoff, gracefully falling back to a top-3 retrieved passages view if all attempts fail.
- **Cached Answers**: `@st.cache_data` keyed by query, scope, and document state hash to prevent redundant API calls and protect quota.
- **Calibrated Distance Thresholds**: Calibrated cosine distance gates (`0.45` for course notes, `0.38` for user uploads) ensure the assistant returns `"I couldn't find this in the selected documents."` on irrelevant or out-of-scope questions instead of hallucinating.
- **Glassmorphic UI**: Contrast-optimized translucent interface over a high-definition background video with expandable source panels, distance metrics, and status badges.

---

## 🏛️ Architecture

The system is structured as modular components under [`src/`](file:///d:/Valentius%20Kryptix/rag-powered-ai-app/src):

- [`src/config.py`](file:///d:/Valentius%20Kryptix/rag-powered-ai-app/src/config.py): Configuration loader reading `config.yaml` and `.env` (`os.environ` → `st.secrets` → `.env`).
- [`src/ingestion/loader.py`](file:///d:/Valentius%20Kryptix/rag-powered-ai-app/src/ingestion/loader.py): Document loading from `data/` and in-memory byte extraction via `load_pdf_bytes()` with `UploadError` validation for scanned, empty, corrupt, or encrypted PDFs.
- [`src/chunking/chunker.py`](file:///d:/Valentius%20Kryptix/rag-powered-ai-app/src/chunking/chunker.py): Boundary-aware chunking splitting at sentences and paragraphs (2,000 chars, 200 overlap).
- [`src/embeddings/embedder.py`](file:///d:/Valentius%20Kryptix/rag-powered-ai-app/src/embeddings/embedder.py): Batch document and query vector generation with Google Gemini (`gemini-embedding-001`), supporting live progress reporting and exponential backoff retry for transient rate limits.
- [`src/vectordb/vector_store.py`](file:///d:/Valentius%20Kryptix/rag-powered-ai-app/src/vectordb/vector_store.py): Manages persistent course notes in ChromaDB and ephemeral session collections (`index_uploaded`, `list_uploaded`, `remove_uploaded`, `clear_uploaded`) with per-session concurrency locks and LRU collection pruning.
- [`src/retrieval/retriever.py`](file:///d:/Valentius%20Kryptix/rag-powered-ai-app/src/retrieval/retriever.py): Top-$k$ similarity search across course notes, uploads, or unified scopes with dual distance threshold filtering.
- [`src/prompts/prompt_templates.py`](file:///d:/Valentius%20Kryptix/rag-powered-ai-app/src/prompts/prompt_templates.py): Document-neutral system instructions, few-shot examples, prompt-injection guards, and citation templates.
- [`src/llm/llm_client.py`](file:///d:/Valentius%20Kryptix/rag-powered-ai-app/src/llm/llm_client.py): Gemini LLM client supporting grounded RAG generation, model failover rotation, upload state hashing for cache invalidation, and graceful degradation.
- [`src/utils/helpers.py`](file:///d:/Valentius%20Kryptix/rag-powered-ai-app/src/utils/helpers.py): Exponential backoff retry decorators and structured application logging.
- [`src/utils/ui.py`](file:///d:/Valentius%20Kryptix/rag-powered-ai-app/src/utils/ui.py): Video background injection (base64 cached) and glassmorphic styling for Streamlit.

---

## 🚀 Setup & Installation

1. **Clone the repository and create a virtual environment:**
   ```bash
   git clone <repo-url>
   cd rag-powered-ai-app
   python -m venv venv
   # Linux / macOS:
   source venv/bin/activate
   # Windows:
   .\venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```
   Add your Gemini API key in `.env`:
   ```ini
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

4. **Ingest course notes into ChromaDB:**
   ```bash
   python ingest.py
   ```
   *(Note: The pre-built course index is already shipped in `chroma_db/`, so the app starts immediately with zero cold-start ingestion lag).*

5. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

---

## 🧪 How to Test

### 1. Test Course Notes Retrieval
Click any of the sample question buttons or type them into the chat input with the **Course notes** scope selected:
1. **"In the W&B versioning example, which command loads version 2 of the model, and what file name is used?"**  
   *Verified answer:* Loads `artifact = wandb.use_artifact('my-model:v2')` and uses file name `model_v2.pth` [Cloud Unit 5.pdf, p.12].
2. **"What bandwidth did the STREAM benchmark show with 75 EC2 instances?"**  
   *Verified answer:* Shows approximately 6,000 MB/s (~6 GB/s) aggregate bandwidth across memory buses [Cloud Unit 2.pdf, p.38].
3. **"In Hadoop, what three components contribute to running a job?"**  
   *Verified answer:* Client, JobTracker, and TaskTracker [Cloud Unit 3.pdf, p.9].

### 2. Test User PDF Upload
1. In the sidebar under **"Chat with your own PDFs"**, drag and drop or browse for a PDF (up to 10 MB).
2. Click **"📥 Index my PDFs"**. Watch the real-time progress bar update as pages are read, chunked, and embedded.
3. Once indexed, the scope automatically switches to **"My uploaded PDFs"**, and an indexed document card appears in the sidebar displaying page and chunk counts.
4. Ask a question specific to your document — the assistant answers concisely with exact citations `[filename.pdf, p.X]` and a **"Answered from your PDF"** badge.
5. Expand the **"Sources"** panel below the answer to inspect retrieved passages, page numbers, and cosine distance scores with **Your upload** tags.
6. Try asking an unrelated or off-topic question — the distance gate automatically flags it and returns `"I couldn't find this in the selected documents."` without hallucinations.
7. Click the trash icon 🗑️ on the document card to remove your file, and verify the collection is automatically purged.

---

## 💻 CLI Usage

You can also run queries directly via the command-line interface:
```bash
# Grounded RAG query over course notes
python query.py "In the W&B versioning example, which command loads version 2 of the model, and what file name is used?"

# Compare RAG response side-by-side with unaugmented Gemini
python query.py "In the W&B versioning example, which command loads version 2 of the model, and what file name is used?" --mode compare

# Inspect retrieved chunks and similarity distance scores
python query.py "What bandwidth did the STREAM benchmark show with 75 EC2 instances?" --mode retrieve
```

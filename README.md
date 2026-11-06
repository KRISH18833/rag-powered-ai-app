# Cloud Notes RAG Assistant

A production-ready Retrieval-Augmented Generation (RAG) assistant built over **Cloud Computing for Machine Learning** course notes, powered by Google Gemini, ChromaDB, and Streamlit. The application retrieves factual passages from 5 modular units, applies distance thresholding to suppress hallucinations, features resilient API retry/failover logic, and renders a glassmorphic web interface with a looping background video.

**Live demo:** [https://cloud-notes-rag.onrender.com](https://cloud-notes-rag.onrender.com)

---

## 🏛️ Architecture

The system is structured as modular components under [`src/`](file:///d:/Valentius%20Kryptix/rag-powered-ai-app/src):

- [`src/config.py`](file:///d:/Valentius%20Kryptix/rag-powered-ai-app/src/config.py): Configuration loader reading `config.yaml` and `.env` (`os.environ` → `st.secrets` → `.env`).
- [`src/ingestion/loader.py`](file:///d:/Valentius%20Kryptix/rag-powered-ai-app/src/ingestion/loader.py): Document loading from `data/` using `pypdf` with character count filtering.
- [`src/chunking/chunker.py`](file:///d:/Valentius%20Kryptix/rag-powered-ai-app/src/chunking/chunker.py): Boundary-aware chunking splitting at sentences and paragraphs (2,000 chars, 200 overlap).
- [`src/embeddings/embedder.py`](file:///d:/Valentius%20Kryptix/rag-powered-ai-app/src/embeddings/embedder.py): Batch document and query vector generation with Google Gemini (`gemini-embedding-001`).
- [`src/vectordb/vector_store.py`](file:///d:/Valentius%20Kryptix/rag-powered-ai-app/src/vectordb/vector_store.py): Persistent vector database storage and cosine indexing via ChromaDB.
- [`src/retrieval/retriever.py`](file:///d:/Valentius%20Kryptix/rag-powered-ai-app/src/retrieval/retriever.py): Top-$k$ similarity search with cosine distance filtering against `max_distance`.
- [`src/prompts/prompt_templates.py`](file:///d:/Valentius%20Kryptix/rag-powered-ai-app/src/prompts/prompt_templates.py): System instructions, grounded RAG prompts, few-shot examples, and source citation templates.
- [`src/llm/llm_client.py`](file:///d:/Valentius%20Kryptix/rag-powered-ai-app/src/llm/llm_client.py): Gemini LLM client supporting grounded RAG generation, plain comparison, and model failovers.
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
   *(Note: If deploying freshly without `chroma_db/`, the Streamlit app will also run ingestion automatically on first startup).*

5. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

---

## ❓ Sample Questions

You can test the RAG assistant with these course-specific questions:
1. **"In the W&B versioning example, which command loads version 2 of the model, and what file name is used?"**  
   *(Found in Cloud Unit 5: answers `run.use_artifact('mnist:v2')` and `model_v2.pth`)*
2. **"What bandwidth did the STREAM benchmark show with 75 EC2 instances?"**  
   *(Found in Cloud Unit 2: answers ~6,000 MB/s or ~6 GB/s across shared memory buses)*
3. **"In Hadoop, what three components contribute to running a job?"**  
   *(Found in Cloud Unit 3: answers Client, JobTracker, and TaskTracker)*

---

## 🎯 Distance Threshold & Fallback

Retrieval is calibrated with a cosine distance threshold (`max_distance: 0.45` in `config.yaml`). If no retrieved chunk has a cosine distance $\le 0.45$, the system suppresses the LLM call entirely and returns the exact fallback line:
> *"I couldn't find this in the course notes."*

This prevents hallucinations and saves API tokens on out-of-scope or irrelevant queries (e.g., *"Who won the 2018 FIFA World Cup?"*).

---

## 💻 CLI Usage

You can also run queries directly via the command-line interface:
```bash
# Grounded RAG query (default)
python query.py "In the W&B versioning example, which command loads version 2 of the model, and what file name is used?"

# Compare RAG response side-by-side with plain Gemini
python query.py "In the W&B versioning example, which command loads version 2 of the model, and what file name is used?" --mode compare

# Inspect retrieved chunks and distance scores
python query.py "What bandwidth did the STREAM benchmark show with 75 EC2 instances?" --mode retrieve
```

---

## ☁️ Deployment on Render

1. Create a new **Web Service** pointing to this repository.
2. Select runtime **Python 3**.
3. Set the **Build Command**:
   ```bash
   pip install -r requirements.txt
   ```
4. Set the **Start Command**:
   ```bash
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```
5. Add the Environment Variable in Render:
   - `GEMINI_API_KEY`: `your_gemini_api_key`

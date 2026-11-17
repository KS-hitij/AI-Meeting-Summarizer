# AI-Meeting-Summarizer

> **AI-powered meeting summarizer with retrieval-augmented generation (RAG).** Upload meeting audio or text files, get structured summaries, and ask natural-language questions against your meeting knowledge base.

A single-file FastAPI backend with LangGraph agents, Celery async workers, Google Gemini for LLM/embedding, OpenAI Whisper for transcription, and ChromaDB + Redis for vector storage and task metadata.

---

## Features

- **Meeting summarization** — Upload an audio (WAV/MP3/MP4) or .txt file; an async worker transcribes it and returns a structured summary.
- **Structured output** — Summaries include `title`, `summary`, `action_items` (owner + task), `open_questions`, and `risks`.
- **Project-based scoping** — Projects group meetings; users are members of projects (comma-separated user IDs), and each project owns its documents.
- **Meets-owned vector store** — Summarized meeting content is embedded and stored in ChromaDB, scoped by `project_id` + `meeting_id`.
- **RAG Q&A** — Ask a question about a project; the system retrieves relevant meeting documents, answers with structured `response` + `evidence`, and runs a judge → improve loop to ensure accuracy.
- **Answer quality judge** — An LLM judge verifies the answer is accurate; if not, the answer is regenerated from retrieved context.
- **Open-source, self-hostable** — Only needs a Redis instance, a database, and Gemini API keys.
- **JWT authentication** — Signup, login, and bearer-token-protected routes backed by PyJWT (HS256) and bcrypt.

---

## Architecture Overview

```
                    ┌─────────────────────────────┐
                    │        Client / API         │
                    │  (upload / signup / login)  │
                    └──────────────┬──────────────┘
                                   │ HTTP (FastAPI + JWT)
                    ┌──────────────▼──────────────┐
                    │    Main API (main.py)       │
                    │  signup/login/projects      │
                    │  /summarize/{project_id}    │
                    │  /rag_query                 │
                    │  /tasks/{task_id}           │
                    └──────────────┬──────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
        ▼                          ▼                          ▼
┌───────────────┐            ┌───────────────┐            ┌───────────────┐
│   Redis       │            │  Celery       │            │  Database     │
│  (broker/back │            │  Workers      │            │  SQLite/Postgres│
│   end for      │            │  (tasks.py)   │            │  (async SQLAchemy)│
│   Celery)      │            │               │            └───────────────┘
└───────────────┘            └───────┬───────┘
                                    │ apply_async
                                    ▼
                          ┌──────────────────┐
                          │ Summarize / RAG   │
                          │  Agent            │
                          │  (LangGraph)      │
                          └───────┬──────────┘
                                  │
        ┌───────────────┬─────────┴─────────┬───────────────┐
        ▼               ▼                   ▼               ▼
   ┌──────────┐    ┌──────────┐        ┌──────────┐    ┌──────────┐
   │ Whisper  │    │ Gemini   │        │  Chroma  │    │ Gemini   │
   │ (transcribe)│   │ (summary/│        │  vector  │    │ embed/   │
   └──────────┘    │  answer)  │        │  DB      │    │ query)   │
                   └──────────┘        └──────────┘    └──────────┘
```

### Two async pipelines

1. **Summarization pipeline** (audio → summary → vector store)
   - Client uploads audio/text → `main.py` saves a temp file and queues `summarize_file`.


---

## Tech Stack

| Concern | Technology |
|---|---|
| Web framework | FastAPI (ASGI, `uvicorn`) |
| DB driver | SQLAlchemy 2.x (async, `create_async_engine`) |
| ORM models | SQLAlchemy 2.x `Mapped`/`mapped_column` |
| Auth | PyJWT (HS256) + bcrypt |
| Async task queue | Celery 5.6 (Redis broker/backend) |
| Vector store | ChromaDB (persistent) |
| Cache / task metadata | Redis 8 |
| LLM | Google Gemini (`google-genai`): gemini-3.1-flash-lite |
| Embeddings | Gemini embedding (`gemini-embedding-2-preview`) |
| Text splitting | `langchain-text-splitters` (tiktoken cl100k) |
| Reranking | `sentence-transformers` CrossEncoder (`bge-reranker-v2-m3`) |
| Transcription | OpenAI Whisper (turbo) |
| Workflow/DL | LangGraph, LangChain |
| Env / secrets | `python-dotenv` |
| Validation / serialization | Pydantic v2, email-validator |

---

## Repository Structure

```
AI-Meeting-Summarizer/
├── main.py                  # FastAPI app, all HTTP routes, auth, upload handling
├── message.py               # LangChain SystemMessage prompts (summarizer/rag/judge/improve)
├── models.py                # Gemini structured-output models (SummarizingAnswer, RagAnswer, etc.)
├── rag_agent.py             # LangGraph RAG agent (retrieve → answer → judge → improve)
├── summarize_agent.py       # LangGraph summarize agent (transcribe → summarize → store)
├── tools.py                 # Tools: transcribe, build_documents, store, search, rerank, embed
├── vectordb.py              # ChromaDB client + collection + Gemini embedding client
├── redis_client.py          # Redis client (broker/backend for Celery + metadata cache)
├── tasks.py                 # Celery tasks: summarize_file, rag
├── util.py                  # JWT helpers + FastAPI exception handlers
├── db/
│   ├── base.py              # SQLAlchemy declarative base


---

## Prerequisites

- Python 3.10+ (the project targets 3.10+ syntax)
- [uv](https://github.com/astral-sh/uv) or `pip`
- Redis server running locally or remotely
- A supported async database (SQLite + `aiosqlite` or Postgres + `asyncpg`)
- Gemini API key

---

## Setup & Installation

```bash
# 1. Clone the repo
git clone https://github.com/KS-hitij/AI-Meeting-Summarizer.git
cd AI-Meeting-Summarizer

# 2. (Recommended) Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
#    Create a .env file and set the variables listed in Environment Variables

# 5. Start Redis (or use a remote Redis instance)
#   e.g. redis-server (native Redis)

# 6. Initialize the database (tables are created automatically on startup)
#    The app calls init_db() during lifespan. To start the server:
uvicorn main:app --factory --reload

# 7. In a separate terminal, start the Celery worker(s)
celery -A tasks.celery worker --loglevel=info

# The API will be available at http://127.0.0.1:8000.
# Interactive docs: Swagger UI at /docs, ReDoc at /redoc.
```

To produce a production build, run `uvicorn main:app --factory` without `--reload` and set `MAX_UPLOAD_SIZE_BYTES` / `AUTH_SECRET` appropriately.

│   ├── database.py          # async engine/session, init_db, get_db dependency
│   ├── user.py              # User ORM model
│   └── project.py           # Project ORM model
├── requirements.txt         # Python dependencies
├── .gitignore
└── uploads/                 # Temporary uploaded files (created at runtime)
```

   - A Celery worker runs the LangGraph `summarize_agent`:
     - Audio → `transcribe_node` (Whisper turbo) → `summarize_node` (structured summary) → `store_in_vector_db_node` (embed + store in ChromaDB).
   - The worker deletes the temp file after finishing.

2. **RAG query pipeline** (question → relevant docs → answer → judge)
   - Client POSTs a question for a project → `main.py` queues `rag`.
   - The `rag_agent` LangGraph:
     - `retrieve_data` — embed the query, search ChromaDB (filtered by `project_id`), cache results in Redis (60 s).
     - `answer_user_query` — structured LLM answer (`response` + `evidence`).
     - `judge_llm_answer` — judge verifies accuracy; if false, fall through to `improve_llm_response` and regenerate.
     - Final answer is the verified/improved response.

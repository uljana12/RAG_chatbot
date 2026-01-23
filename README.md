# Copenhagen IT Job Search Assistant

A production-ready **Retrieval-Augmented Generation (RAG)** platform built with LangChain, helping job seekers find IT positions in Copenhagen, Denmark.

> **Purpose**: AI-powered assistant for Copenhagen IT job search  
> **Architecture**: Modern RAG pipeline with conversation memory  
> **LLM**: Ollama with Llama 3.2 (free, runs locally)

## 🎯 Key Features

| Feature | Technology | Description |
|---------|------------|-------------|
| **RAG Pipeline** | LangChain | Document ingestion, chunking, retrieval with source deduplication |
| **Vector Store** | ChromaDB (Persistent) | Durable storage with singleton client pattern |
| **LLM Backend** | Ollama (Llama 3.2) | Free local LLM, no API key needed |
| **Embeddings** | langchain_ollama | Local embeddings generation |
| **Web UI** | Streamlit | Modern chat interface with Inter font |
| **REST API** | FastAPI | Production API with OpenAPI docs |
| **Containerization** | Docker Compose | One-command deployment |
| **Observability** | Prometheus/Grafana | Metrics and monitoring |

## 📁 Project Structure

```
LLM_Example/
├── app.py                      # Streamlit Web UI
├── Dockerfile                  # Multi-stage production build
├── docker-compose.yml          # Full stack deployment
├── requirements.txt            # Python dependencies
├── .env                        # Configuration
│
├── src/
│   ├── __init__.py
│   ├── ingestion.py           # Document processing & ChromaDB storage
│   ├── rag_chain.py           # RAG chain with conversation memory
│   ├── job_scraper.py         # Copenhagen IT job listings (mock data)
│   └── api.py                 # FastAPI REST API endpoints
│
├── docs/
│   └── ARCHITECTURE.md        # C4 diagrams & technical docs
│
├── monitoring/
│   └── prometheus.yml         # Metrics configuration
│
└── data/
    └── sample_knowledge_base.txt
```

## 🛠️ Setup

### 1. Create Virtual Environment

```bash
cd LLM_Example
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama (Local LLM - FREE)

```bash
# macOS
brew install ollama
brew services start ollama

# Pull the model
ollama pull llama3.2
```

### 4. Configure Environment (Optional)

```bash
cp .env.example .env
```

Default configuration uses Ollama (free, local). No API key needed!

> **Note**: Data is stored persistently in the `chroma_db/` folder.

### 5. Run the Application

**Option A: Local (Development)**
```bash
source venv/bin/activate
streamlit run app.py --server.headless true
```

**Option B: Docker (Production)**
```bash
docker build -t rag-chatbot .
docker run -p 8501:8501 -e OLLAMA_BASE_URL=http://host.docker.internal:11434 rag-chatbot
```

The app will open in your browser at `http://localhost:8501`

## 📖 Usage

### Adding Data

1. **Text Input**: Paste text directly into the sidebar
2. **URLs**: Enter web URLs to scrape and index
3. **File Upload**: Upload PDF, TXT, or MD files

### Chatting

Once you've added data, simply type your questions in the chat input. The chatbot will:
1. Search your knowledge base for relevant information
2. Use the retrieved context to generate an accurate response
3. Show source documents used for the answer

### Example Queries

After loading the sample data:
- "What products does TechCorp offer?"
- "How much does SmartAssist AI cost?"
- "What is the refund policy?"
- "How do I contact customer support?"

## 🔧 Configuration

### Customizing the RAG Pipeline

Edit `src/rag_chain.py` to adjust:
- `temperature`: Controls response creativity (0.0 - 1.0)
- `k`: Number of documents to retrieve (default: 4)
- `DEFAULT_SYSTEM_PROMPT`: The instructions given to the LLM

### Customizing Document Processing

Edit `src/ingestion.py` to adjust:
- `CHUNK_SIZE`: Size of text chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   User Query    │────▶│    Retriever     │────▶│  Vector Store   │
└─────────────────┘     │  (Similarity     │     │   (ChromaDB)    │
                        │   Search)        │     └─────────────────┘
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │ Retrieved Docs   │
                        │ (Top K matches)  │
                        └────────┬─────────┘
                                 │
                                 ▼
┌─────────────────┐     ┌──────────────────┐
│    Response     │◀────│ LLM (Ollama/     │
└─────────────────┘     │ Llama 3.2)       │
                        │ + Context + Query│
                        └──────────────────┘
```

## 📝 API Usage

### REST API (FastAPI)

Start the API server:
```bash
uvicorn src.api:app --reload --port 8000
```

Interactive documentation at: http://localhost:8000/docs

**Example API calls:**

```bash
# Health check
curl http://localhost:8000/health

# Load job data
curl -X POST http://localhost:8000/ingest/jobs

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What Python jobs are available?"}'

# Get metrics
curl http://localhost:8000/metrics
```

### Python SDK

```python
from src.ingestion import ingest_text, load_vector_store
from src.rag_chain import RAGChatbot
from src.job_scraper import get_copenhagen_it_jobs

# Load Copenhagen IT job listings
job_data = get_copenhagen_it_jobs()
ingest_text(job_data, "copenhagen_jobs")

# Or add custom content
ingest_text("Your custom content here...", "my_source")

# Create chatbot instance with vector store
vector_store = load_vector_store()
chatbot = RAGChatbot(vector_store=vector_store)

# Ask questions (sources are automatically deduplicated)
result = chatbot.chat("What Python jobs are available?", include_sources=True)
print(result["answer"])
for source in result["sources"]:
    print(f"  - {source['source']}")
```

## 🐳 Docker Deployment

### Quick Start (Docker Compose)

```bash
# Start all services (Web UI + API + Ollama)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Services:
- **Web UI**: http://localhost:8501
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### With Monitoring

```bash
# Include Prometheus + Grafana
docker-compose --profile monitoring up -d
```

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

## 📊 Architecture Documentation

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for:
- C4 diagrams (Context, Container, Component)
- Data flow diagrams
- Deployment architecture
- Security considerations
- Technology decisions

## 🔧 Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | llama3.2 | Local LLM model |
| `OLLAMA_BASE_URL` | http://localhost:11434 | Ollama service URL |

## 🤝 Contributing

Feel free to submit issues and pull requests!

## 📄 License

MIT License - feel free to use this for learning and personal projects.

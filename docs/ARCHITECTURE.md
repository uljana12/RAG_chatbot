# Architecture Documentation

## C4 Model - System Architecture

This document provides comprehensive architecture documentation using the C4 model (Context, Containers, Components, Code).

---

## Level 1: System Context Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              SYSTEM CONTEXT                                          │
└─────────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────┐          ┌─────────────────────────────────┐          ┌─────────────┐
    │             │          │                                 │          │             │
    │  End User   │─────────▶│   AI Job Search Assistant       │◀────────▶│  LinkedIn   │
    │  (Browser)  │          │   [Software System]             │          │  (External) │
    │             │          │                                 │          │             │
    └─────────────┘          │   Helps users find IT jobs in   │          └─────────────┘
                             │   Copenhagen using RAG and LLM  │
    ┌─────────────┐          │   technology                    │          ┌─────────────┐
    │             │          │                                 │          │             │
    │  Admin      │─────────▶│                                 │◀────────▶│   Ollama    │
    │  (Ingest)   │          │                                 │          │   (LLM)     │
    │             │          └─────────────────────────────────┘          │             │
    └─────────────┘                                                       └─────────────┘
```

### Actors:
- **End User**: Job seekers looking for IT positions in Copenhagen
- **Admin**: System administrators who ingest new job data
- **LinkedIn**: External job listing source (via scraping/API)
- **Ollama**: Local LLM service for inference

---

## Level 2: Container Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              CONTAINER DIAGRAM                                       │
└─────────────────────────────────────────────────────────────────────────────────────┘

                                    ┌─────────────────┐
                                    │    End User     │
                                    │    (Browser)    │
                                    └────────┬────────┘
                                             │ HTTPS
                                             ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│                           AI JOB SEARCH SYSTEM                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                             │   │
│  │  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐     │   │
│  │  │   Web UI        │      │    REST API     │      │   Background    │     │   │
│  │  │   [Streamlit]   │      │   [FastAPI]     │      │   Workers       │     │   │
│  │  │                 │      │                 │      │   [Celery]      │     │   │
│  │  │   Port: 8501    │      │   Port: 8000    │      │                 │     │   │
│  │  └────────┬────────┘      └────────┬────────┘      └────────┬────────┘     │   │
│  │           │                        │                        │              │   │
│  │           └────────────────────────┼────────────────────────┘              │   │
│  │                                    │                                       │   │
│  │                                    ▼                                       │   │
│  │                    ┌───────────────────────────────┐                       │   │
│  │                    │      RAG Engine               │                       │   │
│  │                    │      [Python/LangChain]       │                       │   │
│  │                    │                               │                       │   │
│  │                    │  • Document Processing        │                       │   │
│  │                    │  • Embedding Generation       │                       │   │
│  │                    │  • Retrieval & Ranking        │                       │   │
│  │                    │  • Response Generation        │                       │   │
│  │                    └───────────────┬───────────────┘                       │   │
│  │                                    │                                       │   │
│  │                    ┌───────────────┴───────────────┐                       │   │
│  │                    ▼                               ▼                       │   │
│  │      ┌─────────────────────┐         ┌─────────────────────┐              │   │
│  │      │    Vector Store     │         │    LLM Service      │              │   │
│  │      │    [ChromaDB]       │         │    [Ollama]         │              │   │
│  │      │                     │         │                     │              │   │
│  │      │  Persistent storage │         │  Local: Llama 3.2   │              │   │
│  │      │  for embeddings     │         │  Free, no API key   │              │   │
│  │      └─────────────────────┘         └─────────────────────┘              │   │
│  │                                                                           │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────────┘
```

### Containers:
| Container | Technology | Purpose | Port |
|-----------|------------|---------|------|
| Web UI | Streamlit | User interface for chat | 8501 |
| REST API | FastAPI | Programmatic access | 8000 |
| RAG Engine | Python/LangChain | Core AI logic | - |
| Vector Store | ChromaDB | Embedding storage | - |
| LLM Service | Ollama (Llama 3.2) | Text generation | 11434 |

---

## Level 3: Component Diagram (RAG Engine)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         RAG ENGINE - COMPONENT DIAGRAM                               │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              RAG ENGINE                                              │
│                                                                                      │
│   ┌───────────────────────────────────────────────────────────────────────────┐    │
│   │                        INGESTION PIPELINE                                  │    │
│   │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   │    │
│   │  │  Document   │──▶│    Text     │──▶│  Embedding  │──▶│   Vector    │   │    │
│   │  │  Loaders    │   │  Splitter   │   │  Generator  │   │   Store     │   │    │
│   │  │             │   │             │   │             │   │             │   │    │
│   │  │ • PDF       │   │ Recursive   │   │ Ollama or   │   │ ChromaDB    │   │    │
│   │  │ • Text      │   │ Char Split  │   │ OpenAI      │   │ Persistent  │   │    │
│   │  │ • Web       │   │ 1000/200    │   │             │   │             │   │    │
│   │  │ • Jobs API  │   │             │   │             │   │             │   │    │
│   │  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘   │    │
│   └───────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
│   ┌───────────────────────────────────────────────────────────────────────────┐    │
│   │                        RETRIEVAL PIPELINE                                  │    │
│   │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   │    │
│   │  │   Query     │──▶│  Embedding  │──▶│  Similarity │──▶│   Context   │   │    │
│   │  │  Processing │   │  Query      │   │   Search    │   │  Formatter  │   │    │
│   │  │             │   │             │   │             │   │             │   │    │
│   │  │ Preprocess  │   │ Same model  │   │ Top-K = 4   │   │ Combine     │   │    │
│   │  │ + History   │   │ as indexing │   │ Cosine sim  │   │ documents   │   │    │
│   │  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘   │    │
│   └───────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
│   ┌───────────────────────────────────────────────────────────────────────────┐    │
│   │                        GENERATION PIPELINE                                 │    │
│   │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   │    │
│   │  │   Prompt    │──▶│     LLM     │──▶│   Output    │──▶│  Response   │   │    │
│   │  │  Template   │   │  Inference  │   │  Parser     │   │  + Sources  │   │    │
│   │  │             │   │             │   │             │   │             │   │    │
│   │  │ System +    │   │ Ollama or   │   │ Extract     │   │ Format for  │   │    │
│   │  │ Context +   │   │ OpenAI      │   │ answer      │   │ UI display  │   │    │
│   │  │ History     │   │ temp=0.7    │   │             │   │             │   │    │
│   │  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘   │    │
│   └───────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Level 4: Code Diagram (Key Classes)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              CODE STRUCTURE                                          │
└─────────────────────────────────────────────────────────────────────────────────────┘

src/
├── ingestion.py
│   │
│   ├── get_embeddings() ─────────────────▶ OllamaEmbeddings / OpenAIEmbeddings
│   ├── get_text_splitter() ──────────────▶ RecursiveCharacterTextSplitter
│   ├── load_pdf(path) ───────────────────▶ List[Document]
│   ├── load_text_file(path) ─────────────▶ List[Document]
│   ├── load_from_urls(urls) ─────────────▶ List[Document]
│   ├── process_documents(docs) ──────────▶ List[Document] (chunked)
│   ├── create_vector_store(docs) ────────▶ Chroma
│   ├── load_vector_store() ──────────────▶ Chroma | None
│   ├── add_documents_to_store(docs) ─────▶ Chroma
│   └── clear_vector_store() ─────────────▶ void
│
├── rag_chain.py
│   │
│   ├── get_llm(temperature) ─────────────▶ ChatOllama / ChatOpenAI
│   ├── get_retriever(vector_store, k) ───▶ VectorStoreRetriever
│   ├── format_docs(docs) ────────────────▶ str
│   ├── create_rag_chain(...) ────────────▶ (retrieve_fn, prompt, llm)
│   │
│   └── class RAGChatbot
│       ├── __init__(vector_store, system_prompt, temperature, k)
│       ├── chat(question, include_sources) ──▶ Dict[answer, sources]
│       ├── clear_history() ──────────────────▶ void
│       └── get_history() ────────────────────▶ List[Dict]
│
├── job_scraper.py
│   │
│   ├── @dataclass JobListing
│   │   ├── title, company, location, description
│   │   ├── url, posted_date, job_type, experience_level
│   │   ├── salary, skills
│   │   └── to_text() ────────────────────▶ str
│   │
│   ├── class LinkedInJobScraper
│   │   ├── search_jobs(keywords, location, num_jobs) ──▶ List[JobListing]
│   │   └── get_sample_jobs() ─────────────────────────▶ List[JobListing]
│   │
│   ├── format_jobs_for_ingestion(jobs) ──▶ str
│   └── get_copenhagen_it_jobs() ─────────▶ str
│
└── api.py (NEW)
    │
    ├── /health ──────────────────────────▶ HealthResponse
    ├── /chat ────────────────────────────▶ ChatResponse
    ├── /ingest/text ─────────────────────▶ IngestResponse
    ├── /ingest/jobs ─────────────────────▶ IngestResponse
    └── /metrics ─────────────────────────▶ MetricsResponse
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                               │
└─────────────────────────────────────────────────────────────────────────────────────┘

                              INGESTION FLOW
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Source  │───▶│  Loader  │───▶│ Splitter │───▶│ Embedder │───▶│  Store   │
│  Data    │    │          │    │          │    │          │    │          │
│          │    │ PDF/Web/ │    │ 1000char │    │ Ollama/  │    │ ChromaDB │
│          │    │ Text/Job │    │ chunks   │    │ OpenAI   │    │          │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘


                              QUERY FLOW
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  User    │───▶│  Embed   │───▶│ Retrieve │───▶│ Generate │───▶│ Response │
│  Query   │    │  Query   │    │ Top-K    │    │ Answer   │    │ + Source │
│          │    │          │    │          │    │          │    │          │
│ "Python  │    │ [0.1,    │    │ 4 most   │    │ LLM with │    │ Formatted│
│  jobs?"  │    │  0.3...] │    │ similar  │    │ context  │    │ answer   │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

---

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         DEPLOYMENT ARCHITECTURE                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘

                              KUBERNETES CLUSTER
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                      │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                           INGRESS CONTROLLER                                 │   │
│   │                         (NGINX / Azure App Gateway)                          │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                    │                                    │                           │
│                    ▼                                    ▼                           │
│   ┌────────────────────────────┐      ┌────────────────────────────┐              │
│   │      WEB UI DEPLOYMENT     │      │      API DEPLOYMENT        │              │
│   │  ┌──────────────────────┐  │      │  ┌──────────────────────┐  │              │
│   │  │   Streamlit Pod      │  │      │  │   FastAPI Pod        │  │              │
│   │  │   replicas: 2        │  │      │  │   replicas: 3        │  │              │
│   │  │   cpu: 500m          │  │      │  │   cpu: 1000m         │  │              │
│   │  │   memory: 1Gi        │  │      │  │   memory: 2Gi        │  │              │
│   │  └──────────────────────┘  │      │  └──────────────────────┘  │              │
│   │  Service: ClusterIP:8501   │      │  Service: ClusterIP:8000   │              │
│   └────────────────────────────┘      └────────────────────────────┘              │
│                    │                                    │                           │
│                    └────────────────┬───────────────────┘                           │
│                                     ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                          OLLAMA DEPLOYMENT                                   │   │
│   │  ┌──────────────────────────────────────────────────────────────────────┐   │   │
│   │  │   Ollama Pod (GPU Node)                                              │   │   │
│   │  │   replicas: 1                                                        │   │   │
│   │  │   resources:                                                         │   │   │
│   │  │     requests: { nvidia.com/gpu: 1, memory: 8Gi }                    │   │   │
│   │  │   volumeMounts: /models (PVC for model cache)                        │   │   │
│   │  └──────────────────────────────────────────────────────────────────────┘   │   │
│   │  Service: ClusterIP:11434                                                    │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                               │
│                                     ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                        PERSISTENT STORAGE                                    │   │
│   │  ┌─────────────────────────┐      ┌─────────────────────────┐              │   │
│   │  │  ChromaDB PVC           │      │  Model Cache PVC        │              │   │
│   │  │  storageClass: azure    │      │  storageClass: azure    │              │   │
│   │  │  size: 10Gi             │      │  size: 50Gi             │              │   │
│   │  └─────────────────────────┘      └─────────────────────────┘              │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘

                              AZURE SERVICES
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │
│  │   Azure     │  │   Azure     │  │   Azure     │  │   Azure     │               │
│  │   AKS       │  │   Monitor   │  │   Key Vault │  │   Container │               │
│  │             │  │   (Logs)    │  │   (Secrets) │  │   Registry  │               │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘               │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Security Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           SECURITY LAYERS                                            │
└─────────────────────────────────────────────────────────────────────────────────────┘

Layer 1: Network Security
├── Azure NSG / Firewall rules
├── Private endpoints for Azure services
└── TLS 1.3 for all external communication

Layer 2: Authentication & Authorization
├── Azure AD integration (future)
├── API key authentication for REST API
└── Rate limiting (100 req/min per client)

Layer 3: Application Security
├── Input validation (prompt injection prevention)
├── Output filtering (PII detection)
├── Secrets management (Azure Key Vault / .env)
└── Dependency scanning (pip-audit)

Layer 4: Data Security
├── Encryption at rest (Azure managed keys)
├── Encryption in transit (TLS)
├── Data retention policies
└── Audit logging
```

---

## Monitoring & Observability

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         OBSERVABILITY STACK                                          │
└─────────────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────────┐
                    │           AZURE MONITOR                  │
                    │  ┌───────────┐ ┌───────────┐ ┌────────┐ │
                    │  │  Metrics  │ │   Logs    │ │ Traces │ │
                    │  └─────┬─────┘ └─────┬─────┘ └───┬────┘ │
                    │        │             │           │      │
                    └────────┼─────────────┼───────────┼──────┘
                             │             │           │
        ┌────────────────────┼─────────────┼───────────┼────────────────────┐
        │                    │             │           │                    │
        ▼                    ▼             ▼           ▼                    │
┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐ │
│  Prometheus   │  │   Loki        │  │   Jaeger      │  │   Grafana     │ │
│  (Metrics)    │  │   (Logs)      │  │   (Traces)    │  │  (Dashboards) │ │
└───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘ │
        │                    │                │                            │
        │                    │                │                            │
        ▼                    ▼                ▼                            │
┌─────────────────────────────────────────────────────────────────────────┐│
│                         APPLICATION                                      ││
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    ││
│  │ /metrics    │  │ Structured  │  │ OpenTelemetry│  │ Health      │    ││
│  │ endpoint    │  │ JSON logs   │  │ SDK         │  │ checks      │    ││
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    ││
└─────────────────────────────────────────────────────────────────────────┘│
                                                                           │
Key Metrics:                                                               │
├── request_latency_seconds (histogram)                                    │
├── llm_tokens_used_total (counter)                                        │
├── retrieval_documents_count (histogram)                                  │
├── embedding_latency_seconds (histogram)                                  │
└── active_sessions (gauge)                                                │
```

---

## Technology Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **LLM Framework** | LangChain | Industry standard, excellent abstractions, active community |
| **Vector Store** | ChromaDB | Simple, embedded, good for PoC; Pinecone for production scale |
| **LLM Backend** | Ollama + OpenAI | Flexibility: local for cost/privacy, cloud for scale |
| **Web UI** | Streamlit | Rapid prototyping, good for demos; React for production |
| **API** | FastAPI | High performance, automatic OpenAPI docs, async support |
| **Container** | Docker | Industry standard, multi-stage builds for optimization |
| **Orchestration** | Kubernetes | Enterprise standard, autoscaling, Azure AKS integration |

---

## Future Enhancements

1. **Multi-Agent Architecture**: Add specialized agents for different tasks (search, summarize, compare)
2. **Agentic Workflows**: LangGraph for complex reasoning chains
3. **MCP Integration**: Model Context Protocol for tool integration
4. **Real-time Data**: WebSocket support for streaming responses
5. **A/B Testing**: Multiple prompt variants with evaluation
6. **Fine-tuning Pipeline**: Custom model training on domain data

---

*Document Version: 1.0*  
*Last Updated: January 2026*  
*Author: AI Solutions Architect*

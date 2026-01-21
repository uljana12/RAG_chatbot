"""
FastAPI REST API for the RAG Chatbot
Production-grade API with OpenAPI documentation, health checks, and metrics.
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion import (
    ingest_text,
    ingest_urls,
    load_vector_store,
    clear_vector_store,
    CHROMA_PERSIST_DIRECTORY
)
from src.rag_chain import RAGChatbot
from src.job_scraper import get_copenhagen_it_jobs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Metrics storage (in production, use Prometheus)
metrics = {
    "requests_total": 0,
    "chat_requests": 0,
    "ingest_requests": 0,
    "errors_total": 0,
    "avg_response_time_ms": 0,
    "response_times": []
}

# Global chatbot instance
chatbot_instance: Optional[RAGChatbot] = None


# ============== Pydantic Models ==============

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., example="healthy")
    timestamp: str = Field(..., example="2026-01-21T10:30:00Z")
    version: str = Field(..., example="1.0.0")
    components: Dict[str, str] = Field(..., example={"ollama": "healthy", "vectorstore": "healthy"})


class ChatRequest(BaseModel):
    """Chat request payload."""
    question: str = Field(..., min_length=1, max_length=2000, example="What Python jobs are available?")
    include_sources: bool = Field(default=True, description="Include source documents in response")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation continuity")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What Python developer jobs are available in Copenhagen?",
                "include_sources": True,
                "session_id": "user-123"
            }
        }


class Source(BaseModel):
    """Source document reference."""
    content: str
    source: str


class ChatResponse(BaseModel):
    """Chat response payload."""
    answer: str
    sources: Optional[List[Source]] = None
    response_time_ms: float
    tokens_estimated: Optional[int] = None

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Here are Python developer positions available in Copenhagen...",
                "sources": [{"content": "Job listing excerpt...", "source": "copenhagen_it_jobs"}],
                "response_time_ms": 1523.45,
                "tokens_estimated": 150
            }
        }


class IngestTextRequest(BaseModel):
    """Text ingestion request."""
    text: str = Field(..., min_length=10, max_length=100000)
    source_name: str = Field(default="user_input", max_length=100)

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Company policy documents or any text content to be indexed...",
                "source_name": "company_policies"
            }
        }


class IngestUrlsRequest(BaseModel):
    """URL ingestion request."""
    urls: List[str] = Field(..., min_items=1, max_items=10)

    class Config:
        json_schema_extra = {
            "example": {
                "urls": ["https://example.com/page1", "https://example.com/page2"]
            }
        }


class IngestResponse(BaseModel):
    """Ingestion response."""
    status: str
    message: str
    documents_processed: Optional[int] = None


class MetricsResponse(BaseModel):
    """API metrics response."""
    requests_total: int
    chat_requests: int
    ingest_requests: int
    errors_total: int
    avg_response_time_ms: float
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    timestamp: str


# ============== Lifespan Management ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global chatbot_instance
    app.state.start_time = time.time()
    
    logger.info("Starting RAG Chatbot API...")
    
    # Initialize chatbot if vector store exists
    try:
        if os.path.exists(CHROMA_PERSIST_DIRECTORY):
            vector_store = load_vector_store()
            if vector_store:
                chatbot_instance = RAGChatbot(vector_store=vector_store)
                logger.info("Chatbot initialized with existing vector store")
    except Exception as e:
        logger.warning(f"Could not initialize chatbot: {e}")
    
    yield
    
    logger.info("Shutting down RAG Chatbot API...")


# ============== FastAPI App ==============

app = FastAPI(
    title="Copenhagen IT Job Search API",
    description="""
    ## RAG-powered API for IT Job Search in Copenhagen
    
    This API provides:
    - **Chat**: Ask questions about IT jobs in Copenhagen
    - **Ingest**: Add new documents to the knowledge base
    - **Health**: Check system health and dependencies
    - **Metrics**: Monitor API performance
    
    ### Architecture
    - LangChain for RAG orchestration
    - ChromaDB for vector storage
    - Ollama (Llama 3.2) or OpenAI for LLM
    
    ### Authentication
    API key authentication (X-API-Key header) - coming soon
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Middleware ==============

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and track metrics."""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate response time
    response_time = (time.time() - start_time) * 1000
    
    # Update metrics
    metrics["requests_total"] += 1
    metrics["response_times"].append(response_time)
    if len(metrics["response_times"]) > 100:
        metrics["response_times"] = metrics["response_times"][-100:]
    metrics["avg_response_time_ms"] = sum(metrics["response_times"]) / len(metrics["response_times"])
    
    # Log request
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {response_time:.2f}ms")
    
    # Add response time header
    response.headers["X-Response-Time-Ms"] = str(round(response_time, 2))
    
    return response


# ============== Exception Handlers ==============

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    metrics["errors_total"] += 1
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    metrics["errors_total"] += 1
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if os.getenv("DEBUG") else None,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    )


# ============== Endpoints ==============

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Copenhagen IT Job Search API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Checks the health of:
    - Vector store (ChromaDB)
    - LLM service (Ollama)
    """
    components = {}
    
    # Check vector store
    try:
        if os.path.exists(CHROMA_PERSIST_DIRECTORY):
            components["vectorstore"] = "healthy"
        else:
            components["vectorstore"] = "not_initialized"
    except Exception as e:
        components["vectorstore"] = f"unhealthy: {str(e)}"
    
    # Check Ollama
    try:
        import requests
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            components["ollama"] = "healthy"
        else:
            components["ollama"] = f"unhealthy: status {response.status_code}"
    except Exception as e:
        components["ollama"] = f"unhealthy: {str(e)}"
    
    # Determine overall status
    all_healthy = all(v == "healthy" for v in components.values())
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        timestamp=datetime.utcnow().isoformat() + "Z",
        version="1.0.0",
        components=components
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Send a message to the chatbot.
    
    The chatbot will:
    1. Search the knowledge base for relevant documents
    2. Use the retrieved context to generate an answer
    3. Return the answer with source references
    """
    global chatbot_instance
    
    start_time = time.time()
    metrics["chat_requests"] += 1
    
    # Initialize chatbot if needed
    if chatbot_instance is None:
        try:
            vector_store = load_vector_store()
            if vector_store is None:
                raise HTTPException(
                    status_code=400,
                    detail="Knowledge base is empty. Please ingest some data first using /ingest/jobs or /ingest/text"
                )
            chatbot_instance = RAGChatbot(vector_store=vector_store)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize chatbot: {str(e)}")
    
    try:
        # Get response from chatbot
        result = chatbot_instance.chat(request.question, include_sources=request.include_sources)
        
        response_time = (time.time() - start_time) * 1000
        
        # Format sources
        sources = None
        if request.include_sources and result.get("sources"):
            sources = [Source(content=s["content"], source=s["source"]) for s in result["sources"]]
        
        # Estimate tokens (rough approximation)
        tokens_estimated = len(result["answer"].split()) + len(request.question.split())
        
        return ChatResponse(
            answer=result["answer"],
            sources=sources,
            response_time_ms=round(response_time, 2),
            tokens_estimated=tokens_estimated
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.post("/ingest/text", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_text_endpoint(request: IngestTextRequest, background_tasks: BackgroundTasks):
    """
    Ingest text content into the knowledge base.
    
    The text will be:
    1. Split into chunks
    2. Converted to embeddings
    3. Stored in the vector database
    """
    global chatbot_instance
    
    metrics["ingest_requests"] += 1
    
    try:
        ingest_text(request.text, request.source_name)
        
        # Reinitialize chatbot with new data
        chatbot_instance = None
        
        return IngestResponse(
            status="success",
            message=f"Text ingested successfully from source: {request.source_name}",
            documents_processed=1
        )
        
    except Exception as e:
        logger.error(f"Ingestion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/ingest/urls", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_urls_endpoint(request: IngestUrlsRequest):
    """
    Ingest content from URLs into the knowledge base.
    
    The web pages will be:
    1. Scraped for content
    2. Split into chunks
    3. Converted to embeddings
    4. Stored in the vector database
    """
    global chatbot_instance
    
    metrics["ingest_requests"] += 1
    
    try:
        ingest_urls(request.urls)
        
        # Reinitialize chatbot with new data
        chatbot_instance = None
        
        return IngestResponse(
            status="success",
            message=f"URLs ingested successfully",
            documents_processed=len(request.urls)
        )
        
    except Exception as e:
        logger.error(f"URL ingestion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"URL ingestion failed: {str(e)}")


@app.post("/ingest/jobs", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_jobs():
    """
    Load Copenhagen IT job listings into the knowledge base.
    
    This loads sample job data from major Copenhagen tech companies:
    - Novo Nordisk, Maersk, Spotify, LEGO
    - Microsoft, Danske Bank, Pleo, Unity, etc.
    """
    global chatbot_instance
    
    metrics["ingest_requests"] += 1
    
    try:
        jobs_text = get_copenhagen_it_jobs()
        ingest_text(jobs_text, "copenhagen_it_jobs")
        
        # Reinitialize chatbot with new data
        chatbot_instance = None
        
        return IngestResponse(
            status="success",
            message="Copenhagen IT jobs loaded successfully",
            documents_processed=12
        )
        
    except Exception as e:
        logger.error(f"Jobs ingestion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Jobs ingestion failed: {str(e)}")


@app.delete("/knowledge-base", response_model=IngestResponse, tags=["Ingestion"])
async def clear_knowledge_base():
    """
    Clear all data from the knowledge base.
    
    ⚠️ This action is irreversible!
    """
    global chatbot_instance
    
    try:
        clear_vector_store()
        chatbot_instance = None
        
        return IngestResponse(
            status="success",
            message="Knowledge base cleared successfully"
        )
        
    except Exception as e:
        logger.error(f"Clear error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clear knowledge base: {str(e)}")


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics(request: Request):
    """
    Get API metrics.
    
    Returns:
    - Total requests
    - Chat/Ingest request counts
    - Error count
    - Average response time
    - Uptime
    """
    uptime = time.time() - request.app.state.start_time
    
    return MetricsResponse(
        requests_total=metrics["requests_total"],
        chat_requests=metrics["chat_requests"],
        ingest_requests=metrics["ingest_requests"],
        errors_total=metrics["errors_total"],
        avg_response_time_ms=round(metrics["avg_response_time_ms"], 2),
        uptime_seconds=round(uptime, 2)
    )


# ============== Banking/Finance Use Case Endpoint ==============

@app.post("/ingest/banking-faq", response_model=IngestResponse, tags=["Banking"])
async def ingest_banking_faq():
    """
    Load sample banking FAQ data for Nykredit demo.
    
    This demonstrates the system's applicability to banking/finance use cases:
    - Mortgage information
    - Investment products
    - Customer service automation
    """
    global chatbot_instance
    
    banking_faq = """
NYKREDIT BANKING FAQ - INTERNAL KNOWLEDGE BASE
===============================================

## Mortgage Products

### Realkreditlån (Mortgage Loans)
Q: What types of mortgage loans does Nykredit offer?
A: Nykredit offers several mortgage types:
- FastRente (Fixed Rate): Lock in your interest rate for 1-30 years
- FlexLån (Adjustable Rate): Rate adjusts every 1, 3, or 5 years
- Kort Rente (Short Rate): Lower initial rate with annual adjustments
- F-kort (Ultra-short): Lowest rate, adjusts every 6 months

### Interest Rates
Q: How are mortgage interest rates determined?
A: Rates are based on:
- Bond market prices
- Loan-to-value ratio (LTV)
- Property type and location
- Customer credit profile
Current indicative rates available at nykredit.dk/renter

### Down Payment
Q: What is the minimum down payment for a home purchase?
A: Standard requirements:
- Primary residence: 5% minimum down payment
- Summer house: 25% minimum
- Investment property: 20-40% depending on type
Additional bank loan may cover up to 15% (max 80% LTV with Realkredit + 15% bank loan)

## Investment Products

### Nykredit Invest
Q: What investment options are available?
A: We offer:
- Mutual funds (Investeringsforeninger)
- Pension savings (Pension)
- Stock trading (Aktiehandel)
- Bonds and certificates

### Risk Profiles
Q: How do I choose the right investment profile?
A: We categorize investments by risk:
- Forsigtig (Conservative): 0-30% stocks, stable returns
- Balanceret (Balanced): 30-60% stocks, moderate growth
- Offensiv (Aggressive): 60-100% stocks, higher potential returns

## Digital Services

### Nykredit App
Q: What can I do in the Nykredit mobile app?
A: The app allows you to:
- View accounts and transactions
- Transfer money (MobilePay integrated)
- Apply for loans
- Trade investments
- Contact customer service via chat
- Use NemID/MitID for secure login

### Online Banking
Q: How do I set up online banking?
A: Steps to get started:
1. Register at nykredit.dk with MitID
2. Verify your identity
3. Set up your profile preferences
4. Enable notifications
Support available 24/7 at 70 10 90 00

## Customer Service Automation Use Cases

### Loan Calculator
Input: Property price, down payment, loan term
Output: Monthly payment, total interest, amortization schedule

### Document Processing
Automated extraction from:
- Salary slips (lønsedler)
- Tax returns (årsopgørelse)
- Property valuations (vurdering)

### Chat Support Topics
Common queries handled by AI:
- Account balance inquiries
- Transaction history
- Branch locations and hours
- Rate quotes
- Document upload assistance

## Compliance & Security

### GDPR
All customer data processed according to:
- EU General Data Protection Regulation
- Danish Financial Supervisory Authority requirements
- Internal data governance policies

### Fraud Prevention
AI-powered monitoring for:
- Unusual transaction patterns
- Identity verification
- Anti-money laundering (AML)

---
Last Updated: January 2026
Classification: Internal Use
"""
    
    metrics["ingest_requests"] += 1
    
    try:
        ingest_text(banking_faq, "nykredit_banking_faq")
        
        # Reinitialize chatbot
        chatbot_instance = None
        
        return IngestResponse(
            status="success",
            message="Banking FAQ loaded successfully - demonstrating finance sector applicability",
            documents_processed=1
        )
        
    except Exception as e:
        logger.error(f"Banking FAQ ingestion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Banking FAQ ingestion failed: {str(e)}")


# ============== Main ==============

if __name__ == "__main__":
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

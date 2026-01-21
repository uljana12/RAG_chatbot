"""
RAG Chain Module
Implements the Retrieval-Augmented Generation chain using LangChain.
"""

import os
from typing import List, Dict, Any, Optional
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

from src.ingestion import load_vector_store, get_embeddings, CHROMA_PERSIST_DIRECTORY

load_dotenv()

# Ollama configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# Default system prompt for the chatbot
DEFAULT_SYSTEM_PROMPT = """You are a helpful Copenhagen IT Job Search Assistant. Your role is to help job seekers find relevant IT and Software positions in Copenhagen, Denmark.

Instructions:
- Use the job listings in the context to answer questions accurately
- When listing jobs, include key details: company, title, salary, and required skills
- Format salaries in DKK (Danish Kroner) when available
- Highlight relevant benefits and perks
- If asked about specific skills, filter jobs that require those skills
- Compare positions when asked (salary, benefits, experience level)
- If the context doesn't have relevant jobs, say so honestly
- Be encouraging and helpful to job seekers

When presenting jobs, use this format:
📌 **Job Title** at **Company**
💰 Salary: XXX DKK
🛠️ Skills: skill1, skill2, skill3
📍 Location: Copenhagen, Denmark

Context (Job Listings):
{context}
"""


def get_llm(temperature: float = 0.7):
    """Get the Ollama LLM instance (FREE - runs locally)."""
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature
    )


def get_retriever(vector_store: Optional[Chroma] = None, k: int = 4):
    """Get the retriever from the vector store."""
    if vector_store is None:
        vector_store = load_vector_store()
    
    if vector_store is None:
        raise ValueError("No vector store found. Please ingest some documents first.")
    
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents into a single string."""
    return "\n\n---\n\n".join([
        f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
        for doc in docs
    ])


def create_rag_chain(
    vector_store: Optional[Chroma] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.7,
    k: int = 4
):
    """Create the RAG chain."""
    retriever = get_retriever(vector_store, k)
    llm = get_llm(temperature)
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{question}")
    ])
    
    # Create the chain
    def retrieve_and_format(inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs["question"]
        docs = retriever.invoke(question)
        return {
            **inputs,
            "context": format_docs(docs),
            "source_documents": docs
        }
    
    return retrieve_and_format, prompt, llm


class RAGChatbot:
    """RAG-powered chatbot with conversation memory."""
    
    def __init__(
        self,
        vector_store: Optional[Chroma] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.7,
        k: int = 4
    ):
        self.vector_store = vector_store or load_vector_store()
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.k = k
        self.chat_history: List[BaseMessage] = []
        
        # Initialize components
        self.retrieve_and_format, self.prompt, self.llm = create_rag_chain(
            self.vector_store,
            self.system_prompt,
            self.temperature,
            self.k
        )
    
    def chat(self, question: str, include_sources: bool = False) -> Dict[str, Any]:
        """
        Send a message to the chatbot and get a response.
        
        Args:
            question: The user's question
            include_sources: Whether to include source documents in response
            
        Returns:
            Dict with 'answer' and optionally 'sources'
        """
        # Retrieve relevant documents
        inputs = {"question": question, "chat_history": self.chat_history}
        enriched_inputs = self.retrieve_and_format(inputs)
        
        # Generate response
        messages = self.prompt.format_messages(
            context=enriched_inputs["context"],
            chat_history=self.chat_history,
            question=question
        )
        
        response = self.llm.invoke(messages)
        answer = response.content
        
        # Update chat history
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))
        
        result = {"answer": answer}
        
        if include_sources:
            # Deduplicate sources by content
            seen_content = set()
            unique_sources = []
            for doc in enriched_inputs["source_documents"]:
                content_preview = doc.page_content[:200]
                if content_preview not in seen_content:
                    seen_content.add(content_preview)
                    unique_sources.append({
                        "content": content_preview + "...",
                        "source": doc.metadata.get("source", "Unknown")
                    })
            result["sources"] = unique_sources
        
        return result
    
    def clear_history(self):
        """Clear the conversation history."""
        self.chat_history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get the conversation history as a list of dicts."""
        history = []
        for msg in self.chat_history:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
        return history


def simple_query(question: str, k: int = 4) -> str:
    """
    Simple one-off query without conversation history.
    
    Args:
        question: The question to ask
        k: Number of documents to retrieve
        
    Returns:
        The answer string
    """
    chatbot = RAGChatbot(k=k)
    result = chatbot.chat(question)
    return result["answer"]


if __name__ == "__main__":
    # Example usage
    print("Testing RAG Chain...")
    
    try:
        chatbot = RAGChatbot()
        
        # Test query
        question = "What is LangChain?"
        print(f"\nQuestion: {question}")
        
        result = chatbot.chat(question, include_sources=True)
        print(f"\nAnswer: {result['answer']}")
        
        if result.get('sources'):
            print("\nSources:")
            for source in result['sources']:
                print(f"  - {source['source']}")
                
    except ValueError as e:
        print(f"Error: {e}")
        print("Please run the ingestion script first to add some documents.")

import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.core.config import settings

# Explicitly ensure the API key is mirrored in the environment for the Google GenAI SDK
if settings.GEMINI_API_KEY:
    os.environ["GEMINI_API_KEY"] = settings.GEMINI_API_KEY

class RAGService:
    def __init__(self):
        print("Loading local FAISS database and embedding model...")
        # Re-initialize the exact same local text embedding model used in ingest.py
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Load the saved local index from disk
        self.vector_store = FAISS.load_local(
            settings.VECTOR_STORE_DIR, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Initialize Gemini API cleanly
        # Note: 'api_key' is used here instead of 'google_api_key'
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            api_key=settings.GEMINI_API_KEY,
            temperature=0.2  # Low temperature keeps answers factual and grounded
        )
        print("RAG Engine initialized successfully.")

    def answer_query(self, query: str) -> dict:
        # 1. Semantic Search: Pull top 3 closest matching paragraphs from FAISS
        docs = self.vector_store.similarity_search(query, k=3)
        
        # 2. Extraction: Compile retrieved text and collect individual source filenames
        context_chunks = []
        sources = []
        for doc in docs:
            context_chunks.append(doc.page_content)
            source_path = doc.metadata.get("source", "Unknown Source")
            sources.append(os.path.basename(source_path))
            
        # Join chunks into one unified text block for the context window
        context_str = "\n\n---\n\n".join(context_chunks)
        clean_sources = list(set(sources)) # Deduplicate sources
        
        # 3. Context Injection: Build standard explicit system instructions
        system_prompt = (
            "You are a professional customer support assistant specializing in Bharti AXA Life Insurance.\n"
            "Answer the user's question accurately using ONLY the provided context below. "
            "If you do not know the answer or if it's not explicitly found in the context, say: "
            "'I'm sorry, I cannot find that exact information in the policy documents.'\n"
            "Do not hallucinate, speculate, or make up policies.\n\n"
            f"Context:\n{context_str}"
        )
        
        # 4. Generation: Send structured prompt messages directly to the LLM
        messages = [
            ("system", system_prompt),
            ("human", query)
        ]
        
        response = self.llm.invoke(messages)
        
        return {
            "answer": response.content,
            "sources": clean_sources
        }

# Singleton instance to avoid reloading files on every API hit
rag_engine = RAGService()
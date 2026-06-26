import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.core.config import settings

if settings.GEMINI_API_KEY:
    os.environ["GEMINI_API_KEY"] = settings.GEMINI_API_KEY

class RAGService:
    def __init__(self):
        print("Loading local FAISS database and embedding model...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = FAISS.load_local(
            settings.VECTOR_STORE_DIR, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            api_key=settings.GEMINI_API_KEY,
            temperature=0.2
        )
        
        # NEW: Simple in-memory session manager
        # Structure: { "session_123": [("human", "hi"), ("ai", "hello")] }
        self.sessions = {}
        print("RAG Engine with Conversational Memory initialized.")

    def answer_query(self, query: str, session_id: str = "default") -> dict:
        # 1. Semantic Search (Look up context chunks using the raw query)
        docs = self.vector_store.similarity_search(query, k=3)
        context_str = "\n\n---\n\n".join([doc.page_content for doc in docs])
        clean_sources = list(set([os.path.basename(doc.metadata.get("source", "Unknown")) for doc in docs]))
        
        # 2. Retrieve or initialize historical logs for this specific user session
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        # 3. Structural Prompt Engineering
        system_prompt = (
            "You are a professional customer support assistant specializing in Bharti AXA Life Insurance.\n"
            "Answer the user's question accurately using ONLY the provided context below. "
            "If you do not know the answer or if it's not explicitly found in the context, say: "
            "'I'm sorry, I cannot find that exact information in the policy documents.'\n"
            "Do not hallucinate, speculate, or make up policies.\n\n"
            f"Context:\n{context_str}"
        )
        
        # Assemble the message sequence: System rules -> Past conversation -> New query
        messages = [("system", system_prompt)]
        
        # Append rolling historical context
        for role, text in self.sessions[session_id]:
            messages.append((role, text))
            
        # Append current user prompt
        messages.append(("human", query))
        
        # 4. Generation
        response = self.llm.invoke(messages)
        
        # 5. Memory Storage: Save this turn to lock context for the next follow-up question
        self.sessions[session_id].append(("human", query))
        self.sessions[session_id].append(("ai", response.content))
        
        return {
            "answer": response.content,
            "sources": clean_sources
        }

rag_engine = RAGService()
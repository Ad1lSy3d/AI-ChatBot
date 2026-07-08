import os
import json
import redis
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
        
        # NEW: Hook up Redis to manage persistent chat logs
        self.redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
        print("RAG Engine with Redis Persistent Conversational Memory initialized.")

    def get_session_history(self, session_id: str) -> list:
        """Helper to safely fetch chat conversation lists straight from Redis."""
        try:
            key = f"chat:history:{session_id}"
            history_raw = self.redis_client.get(key)
            if history_raw:
                return json.loads(history_raw)
        except Exception as e:
            print(f"[Redis Memory Warning] Failed to fetch historical logs: {e}")
        return []

    def save_session_history(self, session_id: str, history: list):
        """Helper to save running chat lists safely to your Mac disk layout."""
        try:
            key = f"chat:history:{session_id}"
            self.redis_client.set(key, json.dumps(history))
        except Exception as e:
            print(f"[Redis Memory Warning] Failed to write historical logs: {e}")

    def answer_query(self, query: str, session_id: str = "default") -> dict:
        # 1. Semantic Search (Look up context chunks using the raw query)
        docs = self.vector_store.similarity_search(query, k=3)
        context_str = "\n\n---\n\n".join([doc.page_content for doc in docs])
        clean_sources = list(set([os.path.basename(doc.metadata.get("source", "Unknown")) for doc in docs]))
        
        # 2. REPLACED: Fetch persistent logs directly out of Redis database storage
        session_history = self.get_session_history(session_id)
        
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
        for role, text in session_history:
            messages.append((role, text))
            
        # Append current user prompt
        messages.append(("human", query))
        
        # 4. Generation
        response = self.llm.invoke(messages)
        
        # 5. REPLACED: Update message arrays and write them safely back to Redis
        session_history.append(("human", query))
        session_history.append(("ai", response.content))
        self.save_session_history(session_id, session_history)
        
        return {
            "answer": response.content,
            "sources": clean_sources
        }

    def embed_text(self, text: str) -> list:
        """Exposes the internal embedding service tool directly to endpoints layer."""
        return self.embeddings.embed_query(text)

rag_engine = RAGService()
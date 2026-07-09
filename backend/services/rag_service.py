import os
import json
import redis
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq         # NEW: Groq integration
from langchain_openai import ChatOpenAI     # NEW: OpenRouter integration
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
        
        # 🟢 Tier 1: Primary Model (Gemini)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            api_key=settings.GEMINI_API_KEY,
            temperature=0.2
        )
        
        # 🟡 Tier 2: Secondary Failover (Groq Client initialization)
        groq_key = os.getenv("GROQ_API_KEY") or getattr(settings, "GROQ_API_KEY", None)
        if groq_key and not groq_key.startswith("gsk_YOUR"):
            self.groq_llm = ChatGroq(
                model="llama-3.1-8b-instant",
                groq_api_key=groq_key,
                temperature=0.2
            )
            print("Tier 2 Failover Engine [Groq Llama-3.1] connected.")
        else:
            self.groq_llm = None
            print("Tier 2 Failover [Groq] skipped: No API Key provided.")

        # 🟠 Tier 3: Tertiary Failover (OpenRouter Client initialization)
        or_key = os.getenv("OPENROUTER_API_KEY") or getattr(settings, "OPENROUTER_API_KEY", None)
        if or_key and not or_key.startswith("sk-or-"):
            self.openrouter_llm = ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                model="meta-llama/llama-3.1-8b-instruct:free",
                openai_api_key=or_key,
                temperature=0.2
            )
            print("Tier 3 Failover Engine [OpenRouter Llama-3.1-Free] connected.")
        else:
            self.openrouter_llm = None
            print("Tier 3 Failover [OpenRouter] skipped: No API Key provided.")

        self.redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
        print("RAG Engine with Redis Persistent Memory and Automated Failover Ready.")

    def get_session_history(self, session_id: str) -> list:
        try:
            key = f"chat:history:{session_id}"
            history_raw = self.redis_client.get(key)
            if history_raw:
                return json.loads(history_raw)
        except Exception as e:
            print(f"[Redis Memory Warning] Failed to fetch historical logs: {e}")
        return []

    def save_session_history(self, session_id: str, history: list):
        try:
            key = f"chat:history:{session_id}"
            self.redis_client.set(key, json.dumps(history))
        except Exception as e:
            print(f"[Redis Memory Warning] Failed to write historical logs: {e}")

    def answer_query(self, query: str, session_id: str = "default") -> dict:
        docs = self.vector_store.similarity_search(query, k=3)
        context_str = "\n\n---\n\n".join([doc.page_content for doc in docs])
        clean_sources = list(set([os.path.basename(doc.metadata.get("source", "Unknown")) for doc in docs]))
        
        session_history = self.get_session_history(session_id)
        
        system_prompt = (
            "You are a professional customer support assistant specializing in Bharti AXA Life Insurance.\n"
            "Answer the user's question accurately using ONLY the provided context below. "
            "If you do not know the answer or if it's not explicitly found in the context, say: "
            "'I'm sorry, I cannot find that exact information in the policy documents.'\n"
            "Do not hallucinate, speculate, or make up policies.\n\n"
            f"Context:\n{context_str}"
        )
        
        messages = [("system", system_prompt)]
        for role, text in session_history:
            messages.append((role, text))
        messages.append(("human", query))
        
        answer_content = ""
        
        # =====================================================================
        # AUTOMATED FAILOVER ROUTING LOGIC
        # =====================================================================
        try:
            # 🏁 ATTEMPT TIER 1: Primary Gemini execution
            print("🤖 [TIER 1] Polling primary Gemini-2.5-Flash cluster...")
            response = self.llm.invoke(messages)
            answer_content = response.content
            
        except Exception as tier1_err:
            err_str = str(tier1_err).upper()
            # Catch strict billing quota blocks or standard 429 resource ceiling limits
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "QUOTA" in err_str:
                print("[TIER 1 CRITICAL] Gemini free tier exhausted or rate-limited.")
                
                # 🔄 ATTEMPT TIER 2: Seamless migration to Groq Cloud infrastructure
                if self.groq_llm:
                    try:
                        print("[FAILOVER -> TIER 2] Routing query to Groq Llama-3.1 engine...")
                        response = self.groq_llm.invoke(messages)
                        answer_content = response.content
                        print("⚡ Tier 2 Processing Successful.")
                    except Exception as tier2_err:
                        print(f"[TIER 2 FAILED] Groq endpoint connection dropped: {tier2_err}")
                        # Fallthrough directly to Tier 3 if Groq fails
                        answer_content = self._execute_tier3_fallback(messages)
                else:
                    # No Groq setup, bypass straight to OpenRouter
                    answer_content = self._execute_tier3_fallback(messages)
            else:
                # If it's a structural syntax break instead of a rate limit, raise it to terminal log output
                raise tier1_err

        # Save context to memory bank
        session_history.append(("human", query))
        session_history.append(("ai", answer_content))
        self.save_session_history(session_id, session_history)
        
        return {
            "answer": answer_content,
            "sources": clean_sources
        }

    def _execute_tier3_fallback(self, messages: list) -> str:
        """Private helper to process backup calls via OpenRouter global edge endpoints."""
        if self.openrouter_llm:
            print("[FAILOVER -> TIER 3] Routing query to OpenRouter Llama-3.1-Free cluster...")
            response = self.openrouter_llm.invoke(messages)
            return response.content
        else:
            raise RuntimeError("All configured model tiers exhausted and no valid tertiary fallbacks available.")

    def embed_text(self, text: str) -> list:
        return self.embeddings.embed_query(text)

rag_engine = RAGService()
import json
import redis
import numpy as np
from typing import Optional, Tuple, Dict, Any

class SemanticCacheService:
    def __init__(self, host: str = "localhost", port: int = 6379, threshold: float = 0.94):
        # Establish connection to your local background Redis engine
        self.redis_client = redis.Redis(host=host, port=port, decode_responses=True)
        self.threshold = threshold # 94% meaning similarity cutoff rule
        
    def get_cached_response(self, query_text: str, query_vector: list) -> Optional[Dict[str, Any]]:
        """Scans historical keys to check if an identical meaning has already been computed."""
        try:
            # 1. Grab all active cache keys stored inside Redis
            keys = self.redis_client.keys("cache:data:*")
            if not keys:
                return None
                
            query_vec_np = np.array(query_vector, dtype=np.float32)
            
            # 2. Iterate and check similarities across entries
            for key in keys:
                cache_id = key.split(":")[-1]
                vector_bytes = self.redis_client.get(f"cache:vector:{cache_id}")
                
                if vector_bytes:
                    # Convert the stringified list back into a numpy vector array
                    cached_vec_np = np.array(json.loads(vector_bytes), dtype=np.float32)
                    
                    # Compute Cosine Similarity metrics
                    dot_product = np.dot(query_vec_np, cached_vec_np)
                    norm_q = np.linalg.norm(query_vec_np)
                    norm_c = np.linalg.norm(cached_vec_np)
                    
                    similarity = dot_product / (norm_q * norm_c) if (norm_q and norm_c) else 0.0
                    
                    # If similarity breaches our 94% threshold, we have a semantic hit!
                    if similarity >= self.threshold:
                        raw_data = self.redis_client.get(key)
                        if raw_data:
                            return json.loads(raw_data)
        except Exception as e:
            print(f"[Cache Warning] Failed to parse cache read lookups: {e}")
        return None

    def set_cache_response(self, query_vector: list, answer_text: str, sources: list, audio_base64: str):
        """Saves text and voice assets inside Redis to protect against future cloud token bills."""
        try:
            # Create a unique incrementing sequence index ID
            cache_id = self.redis_client.incr("cache:global:counter")
            
            payload = {
                "answer": answer_text,
                "sources": sources,
                "audio_base64": audio_base64
            }
            
            # Save payload components with a 24-hour expiration safety limit to keep memory fresh
            expire_seconds = 86400 
            
            self.redis_client.setex(f"cache:data:{cache_id}", expire_seconds, json.dumps(payload))
            self.redis_client.setex(f"cache:vector:{cache_id}", expire_seconds, json.dumps(query_vector))
            
            print(f"⛳️ [Cache Save] Successfully cached new semantic insight at ID #{cache_id}")
        except Exception as e:
            print(f"[Cache Warning] Failed to write entry update to Redis storage layout: {e}")

# Initialize our singleton engine instance
semantic_cache = SemanticCacheService()
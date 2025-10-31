from typing import Dict, List
from .embedding_store import EmbeddingStore

class MemoryStore:
    def __init__(self):
        self.sessions: Dict[str, List[str]] = {}
        self.embedder = EmbeddingStore()

    def add_message(self, session_id: str, message: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append(message)
        self.embedder.add(session_id, message)

    def get_messages(self, session_id: str):
        return self.sessions.get(session_id, [])

    def semantic_context(self, session_id: str, query: str):
        return self.embedder.search(session_id, query)

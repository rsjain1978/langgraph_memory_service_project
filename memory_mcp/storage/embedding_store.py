import faiss
import numpy as np
from openai import OpenAI

class EmbeddingStore:
    def __init__(self, dim=1536):
        self.client = OpenAI()
        self.index = faiss.IndexFlatL2(dim)
        self.vectors = []
        self.metadata = []

    def _embed(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def add(self, session_id: str, message: str):
        vec = self._embed(message)
        self.index.add(np.array([vec]))
        self.vectors.append(vec)
        self.metadata.append({"session_id": session_id, "message": message})

    def search(self, session_id: str, query: str, top_k=5):
        if len(self.vectors) == 0:
            return []
        vec = self._embed(query)
        D, I = self.index.search(np.array([vec]), top_k)
        results = []
        for i in I[0]:
            if i < len(self.metadata):
                md = self.metadata[i]
                if md["session_id"] == session_id:
                    results.append(md["message"])
        return results

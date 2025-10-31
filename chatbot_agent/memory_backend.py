import httpx
from langmem import MemoryBackend

class RemoteMemory(MemoryBackend):
    def __init__(self, base_url="http://localhost:8001"):
        self.base_url = base_url

    async def save(self, session_id: str, message: str):
        async with httpx.AsyncClient() as client:
            await client.post(f"{self.base_url}/save", json={
                "session_id": session_id,
                "message": message
            })

    async def context(self, session_id: str, query: str):
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{self.base_url}/context/{session_id}",
                                   params={"query": query})
            return res.json().get("context", [])

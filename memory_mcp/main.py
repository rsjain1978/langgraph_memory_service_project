from fastapi import FastAPI, Request
from storage.memory_store import MemoryStore

app = FastAPI(title="Memory MCP Service (OpenAI Embeddings)")
store = MemoryStore()

@app.post("/save")
async def save_memory(req: Request):
    data = await req.json()
    store.add_message(data["session_id"], data["message"])
    return {"status": "saved"}

@app.get("/context/{session_id}")
async def get_context(session_id: str, query: str):
    context = store.semantic_context(session_id, query)
    return {"context": context}

@app.get("/load/{session_id}")
async def load_all(session_id: str):
    return {"messages": store.get_messages(session_id)}

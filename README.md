# LangGraph Memory Service Project

A modular conversational AI system featuring a **LangGraph-based chatbot agent** with intelligent triage capabilities and a **semantic memory microservice** powered by OpenAI embeddings and FAISS vector search.

## 🏗️ Architecture

This project consists of two main components:

1. **Memory MCP Service** (`memory_mcp/`) - A FastAPI microservice that provides semantic memory storage and retrieval
2. **Chatbot Agent** (`chatbot_agent/`) - A LangGraph-based conversational agent with triage and response capabilities

```
┌─────────────────────────────────────────────────────┐
│                  Chatbot Agent                      │
│  ┌──────────────┐       ┌──────────────┐          │
│  │    Triage    │──────▶│   Respond    │          │
│  │    Agent     │       │    Agent     │          │
│  └──────┬───────┘       └──────┬───────┘          │
│         │                      │                   │
│         └──────────┬───────────┘                   │
│                    │                               │
│            ┌───────▼────────┐                      │
│            │ Remote Memory  │                      │
│            │    Backend     │                      │
│            └───────┬────────┘                      │
└────────────────────┼──────────────────────────────┘
                     │ HTTP
                     │
┌────────────────────▼──────────────────────────────┐
│            Memory MCP Service                     │
│  ┌─────────────────────────────────────────┐     │
│  │           Memory Store                  │     │
│  │  ┌────────────────────────────────┐    │     │
│  │  │     Embedding Store (FAISS)    │    │     │
│  │  │   - OpenAI text-embedding      │    │     │
│  │  │   - Vector similarity search   │    │     │
│  │  └────────────────────────────────┘    │     │
│  └─────────────────────────────────────────┘     │
└───────────────────────────────────────────────────┘
```

## ✨ Features

### Memory MCP Service
- **Semantic Memory Storage**: Stores conversation messages with vector embeddings
- **Session Isolation**: Maintains separate memory contexts for different users/sessions
- **Semantic Search**: Retrieves relevant context based on semantic similarity (not just keywords)
- **RESTful API**: Easy-to-use HTTP endpoints for memory operations
- **Vector Search**: Uses FAISS for efficient similarity search
- **OpenAI Embeddings**: Leverages `text-embedding-3-small` for high-quality embeddings

### Chatbot Agent
- **Intelligent Triage**: Decides whether to respond based on conversation context
- **Context-Aware Responses**: Uses semantic memory to inform responses
- **State Graph Workflow**: Built with LangGraph for clear conversation flow
- **Session Management**: Maintains conversation context across interactions
- **GPT-4o Integration**: Powered by OpenAI's GPT-4o-mini model
- **Asynchronous Processing**: Efficient async/await architecture

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key
- Git (for cloning the repository)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd langgraph_memory_service_project
```

2. **Set up the Memory MCP Service**
```bash
cd memory_mcp
pip install -r requirements.txt
```

3. **Set up the Chatbot Agent**
```bash
cd ../chatbot_agent
pip install -r requirements.txt
```

4. **Configure environment variables**

Create a `.env` file or set environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

### Running the Application

#### Step 1: Start the Memory MCP Service

In one terminal:
```bash
cd memory_mcp
uvicorn main:app --port 8001 --reload
```

The memory service will be available at `http://localhost:8001`

#### Step 2: Start the Chatbot Agent

In another terminal:
```bash
cd chatbot_agent
python main.py
```

#### Step 3: Chat!

Once both services are running, you can interact with the chatbot:
```
You: Hello! My name is Alice.
Bot: Hi Alice! Nice to meet you. How can I help you today?

You: What's the weather like?
Bot: (no response)

You: Can you remember my name?
Bot: Yes, of course! Your name is Alice.

Type 'exit' or 'quit' to end the conversation.
```

## 📁 Project Structure

```
langgraph_memory_service_project/
│
├── memory_mcp/                    # Memory microservice
│   ├── main.py                    # FastAPI application entry point
│   ├── requirements.txt           # Service dependencies
│   └── storage/
│       ├── memory_store.py        # Memory storage manager
│       └── embedding_store.py     # Vector embedding & FAISS search
│
└── chatbot_agent/                 # Conversational agent
    ├── main.py                    # Agent entry point & LangGraph workflow
    ├── triage_agent.py            # Decision logic for responding
    ├── respond_agent.py           # Response generation logic
    ├── memory_backend.py          # HTTP client for memory service
    └── requirements.txt           # Agent dependencies
```

## 🔧 API Documentation

### Memory MCP Service Endpoints

#### POST `/save`
Save a message to memory for a specific session.

**Request:**
```json
{
  "session_id": "user123",
  "message": "Hello, I'm learning about AI!"
}
```

**Response:**
```json
{
  "status": "saved"
}
```

#### GET `/context/{session_id}`
Retrieve semantically relevant context for a query.

**Parameters:**
- `session_id` (path): The session identifier
- `query` (query string): The search query

**Example:**
```
GET /context/user123?query=What did I say about AI?
```

**Response:**
```json
{
  "context": [
    "Hello, I'm learning about AI!",
    "I find machine learning fascinating"
  ]
}
```

#### GET `/load/{session_id}`
Retrieve all messages for a session.

**Example:**
```
GET /load/user123
```

**Response:**
```json
{
  "messages": [
    "User: Hello, I'm learning about AI!",
    "Bot: That's great! AI is a fascinating field..."
  ]
}
```

## 🧠 How It Works

### Conversation Flow

1. **User Input**: User types a message
2. **Triage**: The triage agent retrieves relevant context and decides if the bot should respond
3. **Response Generation** (if approved):
   - Retrieve semantic context related to the user's message
   - Generate a contextually-aware response using GPT-4o-mini
   - Save both the user message and bot response to memory
4. **Output**: Display the bot's response to the user

### Memory System

The memory system uses **semantic embeddings** to store and retrieve conversation context:

1. **Storage**: When a message is saved:
   - Generate an embedding vector using OpenAI's `text-embedding-3-small` model
   - Store the vector in FAISS index
   - Associate it with session metadata

2. **Retrieval**: When context is needed:
   - Generate an embedding for the query
   - Perform similarity search in FAISS index
   - Return the top-k most similar messages from the same session

This approach allows the bot to:
- Remember important context from earlier in the conversation
- Find relevant information even when exact keywords don't match
- Maintain separate memory contexts for different users

### Triage Logic

The triage agent helps manage conversation flow by:
- Analyzing user input and conversation history
- Determining relevance and engagement level
- Preventing unnecessary responses to off-topic or rhetorical messages
- Reducing API costs by filtering non-actionable inputs

## 📦 Dependencies

### Memory MCP Service
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `openai` - OpenAI API client
- `faiss-cpu` - Vector similarity search

### Chatbot Agent
- `langgraph` - State graph orchestration
- `httpx` - Async HTTP client
- `openai` - OpenAI API client

## ⚙️ Configuration

### Memory Service Configuration

Default settings in `memory_backend.py`:
- **Base URL**: `http://localhost:8001`
- **Port**: 8001

To change the memory service URL:
```python
memory = RemoteMemory(base_url="http://your-server:port")
```

### Embedding Configuration

Default settings in `embedding_store.py`:
- **Model**: `text-embedding-3-small`
- **Dimension**: 1536
- **Top-K Results**: 5

### LLM Configuration

Default settings in `triage_agent.py` and `respond_agent.py`:
- **Triage Model**: `gpt-4o-mini` (temperature=0)
- **Response Model**: `gpt-4o-mini` (temperature=0.7)

## 🔍 Example Use Cases

1. **Customer Support Bot**: Maintain context across multi-turn conversations
2. **Personal Assistant**: Remember user preferences and past interactions
3. **Educational Tutor**: Track learning progress and adapt explanations
4. **Research Assistant**: Recall information from long conversations
5. **Meeting Facilitator**: Remember action items and decisions

## 🛠️ Extending the Project

### Add New Agents

Create a new agent in `chatbot_agent/`:
```python
from openai import AsyncOpenAI

class CustomAgent:
    def __init__(self, memory):
        self.client = AsyncOpenAI()
        self.memory = memory
    
    async def process(self, session_id: str, message: str):
        context = await self.memory.context(session_id, message)
        # Your logic here
        return result
```

### Add Persistent Storage

Currently, the memory is in-memory only. To add persistence:

1. Replace `self.sessions` dict with a database (SQLite, PostgreSQL, etc.)
2. Serialize/deserialize FAISS index to disk
3. Implement save/load methods in `MemoryStore`

Example with SQLite:
```python
import sqlite3

class PersistentMemoryStore(MemoryStore):
    def __init__(self, db_path="memory.db"):
        super().__init__()
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
```

### Use Different Embedding Models

To use a different embedding provider:

1. Update `embedding_store.py`:
```python
from sentence_transformers import SentenceTransformer

class EmbeddingStore:
    def __init__(self, dim=384):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.IndexFlatL2(dim)
        # ... rest of implementation
```

### Deploy to Production

For production deployment:

1. **Memory Service**:
   - Use a production ASGI server (e.g., Gunicorn with Uvicorn workers)
   - Add authentication/authorization
   - Implement rate limiting
   - Add logging and monitoring
   - Use Redis or a database for persistence

2. **Chatbot Agent**:
   - Add error handling and retries
   - Implement connection pooling
   - Add health checks
   - Configure timeout settings
   - Set up logging

## 🐛 Troubleshooting

### Common Issues

**Issue**: Memory service connection refused
- **Solution**: Ensure the memory service is running on port 8001

**Issue**: OpenAI API errors
- **Solution**: Verify your `OPENAI_API_KEY` environment variable is set correctly

**Issue**: FAISS installation fails
- **Solution**: Try `pip install faiss-cpu` instead of `faiss-gpu` if you don't have CUDA

**Issue**: Bot doesn't respond
- **Solution**: The triage agent may have decided not to respond. Check the triage logic or adjust the prompt.

## 📝 License

This project is provided as-is for educational and development purposes.

## 🤝 Contributing

Contributions are welcome! Some ideas:
- Add support for multiple embedding models
- Implement persistent storage
- Add web UI for the chatbot
- Enhance triage logic with more sophisticated criteria
- Add conversation summarization
- Implement memory pruning/archiving strategies
- Add metrics and monitoring

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

**Built with ❤️ using LangGraph, OpenAI, and FAISS**


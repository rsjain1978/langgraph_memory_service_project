import asyncio
from langgraph.graph import StateGraph, END
from triage_agent import TriageAgent
from respond_agent import RespondAgent
from memory_backend import RemoteMemory

memory = RemoteMemory()
triage = TriageAgent(memory)
responder = RespondAgent(memory)

graph = StateGraph()

@graph.node()
async def triage_node(state):
    user_message = state["message"]
    session_id = state["session"]
    should_respond = await triage.should_respond(session_id, user_message)
    state["should_respond"] = should_respond
    return state

@graph.node()
async def respond_node(state):
    if state["should_respond"]:
        reply = await responder.reply(state["session"], state["message"])
        state["reply"] = reply
    else:
        state["reply"] = "(no response)"
    return state

graph.edge("triage_node", "respond_node")
graph.edge("respond_node", END)

async def run_chat():
    session = "user123"
    while True:
        msg = input("You: ")
        if msg.lower() in ["exit", "quit"]:
            break
        result = await graph.run({"session": session, "message": msg})
        print("Bot:", result["reply"])

if __name__ == "__main__":
    asyncio.run(run_chat())

from openai import AsyncOpenAI

class TriageAgent:
    def __init__(self, memory):
        self.client = AsyncOpenAI()
        self.memory = memory

    async def should_respond(self, session_id: str, user_message: str):
        context = await self.memory.context(session_id, user_message)
        joined_context = "\n".join(context)

        prompt = f"""
You are a triage agent. You decide if a chatbot should respond to a message.

Use the past context to infer relevance and engagement:
{joined_context}

User message: "{user_message}"

Respond only with "yes" or "no".
"""

        completion = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        decision = completion.choices[0].message.content.strip().lower()
        await self.memory.save(session_id, f"Triage decision: {decision}")
        return decision == "yes"

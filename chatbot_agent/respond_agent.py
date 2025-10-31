from openai import AsyncOpenAI

class RespondAgent:
    def __init__(self, memory):
        self.client = AsyncOpenAI()
        self.memory = memory

    async def reply(self, session_id: str, user_message: str):
        context = await self.memory.context(session_id, user_message)
        joined_context = "\n".join(context)

        prompt = f"""
You are a helpful, conversational assistant. Refer to relevant context if needed.

Context:
{joined_context}

User: {user_message}
Bot:
"""

        completion = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        response = completion.choices[0].message.content.strip()
        await self.memory.save(session_id, f"User: {user_message}")
        await self.memory.save(session_id, f"Bot: {response}")
        return response

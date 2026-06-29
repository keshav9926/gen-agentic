# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║       AI STUDY BUDDY — STEP 3: Giving It a Memory               ║
║                   (Conversation Memory)                         ║
╚══════════════════════════════════════════════════════════════════╝

WHAT WE'RE BUILDING TODAY:
  A real chatbot that remembers the full conversation.
  You can say "explain more" or "give me an example of that"
  and it will know what "that" refers to.

PROBLEM WITH STEP 2:
  Every .invoke() call is stateless — completely independent.
  The LLM has NO idea what you said 2 messages ago.

  You: "Explain transformers"       → AI explains
  You: "Now give me an example"     → AI: "Example of what?" 😕

  This is because LLMs are STATELESS by design.
  They only see what's in the current context window.

CONCEPT: Context Window
─────────────────────────
  An LLM can only "see" a fixed amount of text at one time.
  This is the context window (measured in tokens):

  GPT-3.5   →  4,096 tokens  (~3,000 words)
  GPT-4     → 128,000 tokens (~100,000 words)
  Gemini 2  →   1M+ tokens

  MEMORY = manually stuffing past messages into the context window
  before each new call. That's all it is. No magic.

CONCEPT: Types of Memory
──────────────────────────

  1. ConversationBufferMemory
     → Store ALL messages verbatim
     → Problem: context window fills up fast in long conversations

  2. ConversationBufferWindowMemory(k=5)
     → Store only the last k turns
     → Cheap, but forgets old context

  3. ConversationSummaryMemory
     → Summarize old conversation as it grows
     → "Before we talked about X and Y. Now the user asks Z."
     → Smart but requires extra LLM calls to summarize

  4. VectorStoreMemory (coming in Step 5 - RAG)
     → Embed all past messages into a vector DB
     → Retrieve relevant past messages by similarity
     → Most powerful, needed for very long sessions
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
parser = StrOutputParser()


# ─────────────────────────────────────────────────────────────────
# LESSON 3A: Manual Memory — What's REALLY happening under the hood
# ─────────────────────────────────────────────────────────────────
# Before using LangChain's memory classes, understand the raw mechanism:
# Every call includes the FULL conversation history as messages.
print("=" * 60)
print("LESSON 3A: Manual Memory (the raw truth)")
print("=" * 60)

# Simulate what memory actually does
conversation_history = [
    SystemMessage(content="You are a Gen AI tutor. Be concise.")
]

def chat_manually(user_input: str) -> str:
    """Manually manage history — this is what all memory classes do internally."""
    global conversation_history

    # 1. Append user message to history
    conversation_history.append(HumanMessage(content=user_input))

    # 2. Send the ENTIRE history to the LLM
    response = llm.invoke(conversation_history)

    # 3. Append AI response to history
    conversation_history.append(AIMessage(content=response.content))

    return response.content

print(chat_manually("My name is Keshav and I'm learning Gen AI."))
print()
print(chat_manually("What's my name?"))  # It knows! Because history is included.
print()
print(chat_manually("What topic am I learning?"))  # Still knows!

# See what the history looks like
print(f"\nHistory has {len(conversation_history)} messages")


# ─────────────────────────────────────────────────────────────────
# LESSON 3B: LangChain Memory — The Elegant Way
# ─────────────────────────────────────────────────────────────────
# ChatMessageHistory stores messages.
# RunnableWithMessageHistory wraps a chain to auto-inject history.
print("\n" + "=" * 60)
print("LESSON 3B: LangChain's Memory Abstraction")
print("=" * 60)

# The prompt has a special {chat_history} placeholder for past messages
tutor_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are Buddy, a friendly Gen AI tutor. "
        "You remember everything the student tells you. "
        "Be encouraging and concise."
    )),
    # MessagesPlaceholder is where past messages get injected automatically
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

chain = tutor_prompt | llm | parser

# In-memory store: session_id → ChatMessageHistory
# In production: swap this with Redis, DynamoDB, etc.
session_store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Return (or create) the history for a given session."""
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

# Wrap the chain with automatic history management
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,       # function that returns history for a session
    input_messages_key="input",
    history_messages_key="chat_history",
)


# ─────────────────────────────────────────────────────────────────
# LESSON 3C: Multi-Session Memory
# ─────────────────────────────────────────────────────────────────
# The session_id is key — different IDs = different memories
# This is how ChatGPT keeps your conversations separate
print("\n" + "=" * 60)
print("LESSON 3C: Multi-Session (two students, separate memories)")
print("=" * 60)

def student_chat(session_id: str, message: str) -> str:
    """Chat with session isolation."""
    config = {"configurable": {"session_id": session_id}}
    return chain_with_memory.invoke({"input": message}, config=config)

# Student 1: Keshav
print("--- Keshav's session ---")
print(student_chat("keshav_001", "Hi! I'm Keshav. I'm stuck on transformers."))
print()
print(student_chat("keshav_001", "Specifically the attention mechanism. Help?"))
print()
print(student_chat("keshav_001", "What was I stuck on?"))  # Should remember!

# Student 2: different session, different memory
print("\n--- Priya's session ---")
print(student_chat("priya_002", "Hi! I'm Priya, learning about RAG."))
print()
print(student_chat("priya_002", "What's my name?"))  # Knows it's Priya, not Keshav


# ─────────────────────────────────────────────────────────────────
# LESSON 3D: Window Memory — Forget old messages (save tokens)
# ─────────────────────────────────────────────────────────────────
# For long conversations, we don't want to send 200 messages each time.
# Keep only the last k exchanges.
print("\n" + "=" * 60)
print("LESSON 3D: Window Memory (remember last k messages only)")
print("=" * 60)

class WindowedChatHistory(ChatMessageHistory):
    """Only keep the last k messages to avoid context overflow."""
    max_messages: int = 10

    def add_message(self, message):
        super().add_message(message)
        # Trim to last max_messages (keep the most recent)
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]


windowed_store = {}

def get_windowed_history(session_id: str):
    if session_id not in windowed_store:
        windowed_store[session_id] = WindowedChatHistory(max_messages=6)
    return windowed_store[session_id]

windowed_chain = RunnableWithMessageHistory(
    chain,
    get_windowed_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

config = {"configurable": {"session_id": "demo"}}
windowed_chain.invoke({"input": "I'm learning RAG. Start with basics."}, config=config)
windowed_chain.invoke({"input": "Explain chunking."}, config=config)
windowed_chain.invoke({"input": "Now embeddings."}, config=config)
windowed_chain.invoke({"input": "And vector databases."}, config=config)

session_history = get_windowed_history("demo")
print(f"Messages in history: {len(session_history.messages)} (capped at 6)")


# ─────────────────────────────────────────────────────────────────
# PUTTING IT TOGETHER: Study Buddy v3 — Full Chatbot
# ─────────────────────────────────────────────────────────────────
def run_study_buddy_chat():
    """
    Step 3 version: Full interactive chatbot with persistent memory.
    Run this to have a real conversation with your Study Buddy.
    """
    print("\n" + "=" * 60)
    print("STUDY BUDDY v3 — Interactive Chatbot")
    print("Type 'quit' to exit")
    print("=" * 60)

    session_id = "interactive_session"
    config = {"configurable": {"session_id": session_id}}

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Buddy: See you next time! Keep learning! 🚀")
            break
        if not user_input:
            continue

        response = chain_with_memory.invoke({"input": user_input}, config=config)
        print(f"\nBuddy: {response}")

        # Show how many messages are in memory
        history = get_session_history(session_id)
        print(f"  [Memory: {len(history.messages)} messages stored]")


if __name__ == "__main__":
    run_study_buddy_chat()


"""
WHAT YOU LEARNED:
  ✅ LLMs are stateless — memory is just injecting past messages
  ✅ ChatMessageHistory stores messages per session
  ✅ RunnableWithMessageHistory → auto-manages history injection
  ✅ session_id isolates conversations (like ChatGPT's separate chats)
  ✅ Window memory caps history to save tokens and cost

  INTERVIEW INSIGHT:
    "How do you build a multi-user chatbot with separate memory?"
    → session_id maps to a separate ChatMessageHistory per user
    → In production, store history in Redis/DynamoDB, not in-memory dict
    → Use ConversationSummaryMemory for very long sessions

NEXT → step_04_rag.py
  PROBLEM: Our tutor only knows what LLaMA-3 was trained on.
  What if we want it to answer from OUR notes, OUR PDFs, OUR docs?
  → RAG (Retrieval Augmented Generation) solves this.
"""

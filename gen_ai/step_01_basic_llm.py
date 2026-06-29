# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║          AI STUDY BUDDY — STEP 1: The Brain (LLM Basics)        ║
╚══════════════════════════════════════════════════════════════════╝

WHAT WE'RE BUILDING TODAY:
  A simple "ask any question → get an answer" function.
  Just like Google, but it *reasons*, not just retrieves.

CONCEPT: What is an LLM?
─────────────────────────
  An LLM (Large Language Model) is a neural network trained on
  hundreds of billions of text tokens to predict "what comes next."

  During training it saw: "The capital of France is ___"
  millions of times → it learned "Paris" is very likely next.

  At inference (runtime), it does this one token at a time:
    "What" → "is" → "the" → "capital" → "of" → "India" → "?" → "New" → "Delhi"

CONCEPT: What is a Token?
──────────────────────────
  LLMs don't see words — they see tokens (subword chunks).
  "playing" → ["play", "ing"]    (2 tokens)
  "Keshav"  → ["K", "esh", "av"] (3 tokens)
  Rule of thumb: 1 token ≈ 0.75 words

  Why does this matter?
  → Models have a max token limit (context window)
  → You pay per token with paid APIs
  → Longer prompts = slower + more expensive

CONCEPT: Temperature
─────────────────────
  Controls how "creative" or "random" the model is.

  temperature=0   → Always picks the MOST likely next token (deterministic)
                    "2+2=?" → always "4"
  temperature=0.7 → Picks from top candidates (balanced)
                    Good for conversations, explanations
  temperature=1.5 → Very random, picks unlikely tokens
                    Good for creative writing, brainstorming

  Think of it like an exam vs. a brainstorm session.
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# ─────────────────────────────────────────────────────────────────
# THE MODEL — Your AI Study Buddy's brain
# ─────────────────────────────────────────────────────────────────
# ChatGroq wraps the Groq API. Groq runs LLaMA-3 at extreme speeds.
# LangChain's abstraction means swapping to OpenAI/Gemini = 1 line change.
#
# temperature=0 → we want factual, consistent study answers
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


# ─────────────────────────────────────────────────────────────────
# LESSON 1A: Basic invoke — the simplest possible call
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("LESSON 1A: Basic Question → Answer")
print("=" * 60)

# .invoke() sends a message and waits for the full response.
# Returns an AIMessage object — NOT a raw string.
response = llm.invoke("What is a neural network? Explain in 3 sentences.")

print(f"Type:    {type(response)}")       # <class 'langchain_core.messages.ai.AIMessage'>
print(f"Answer:  {response.content}")     # The text we care about
print(f"Tokens:  {response.response_metadata.get('usage', {})}")  # token usage


# ─────────────────────────────────────────────────────────────────
# LESSON 1B: System + Human messages — role-based prompting
# ─────────────────────────────────────────────────────────────────
# Every chat LLM understands 3 roles:
#   SystemMessage  → instructions/persona (not from the user)
#   HumanMessage   → the user's actual question
#   AIMessage      → the model's response (used in history)
#
# The system message is your "briefing" to the AI before it speaks.
print("\n" + "=" * 60)
print("LESSON 1B: System Prompt = AI's Persona")
print("=" * 60)

messages = [
    SystemMessage(content=(
        "You are an expert Gen AI tutor. "
        "Explain concepts simply, always use an analogy, "
        "and end with one 'test your understanding' question."
    )),
    HumanMessage(content="Explain what an embedding is."),
]

response = llm.invoke(messages)
print(response.content)


# ─────────────────────────────────────────────────────────────────
# LESSON 1C: Streaming — word by word output (great UX)
# ─────────────────────────────────────────────────────────────────
# .stream() returns chunks as they are generated, just like ChatGPT.
# Without streaming, user waits ~5 seconds, then sees everything at once.
# With streaming, they start reading immediately.
print("\n" + "=" * 60)
print("LESSON 1C: Streaming Output (token by token)")
print("=" * 60)

print("Answer: ", end="", flush=True)
for chunk in llm.stream("What is backpropagation? Be brief."):
    print(chunk.content, end="", flush=True)  # print each token as it arrives
print()  # newline at end


# ─────────────────────────────────────────────────────────────────
# LESSON 1D: Temperature experiment
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("LESSON 1D: Temperature Effect")
print("=" * 60)

question = "Give me a creative name for an AI assistant."

for temp in [0, 0.7, 1.5]:
    creative_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=temp)
    ans = creative_llm.invoke(question).content.strip()
    print(f"temp={temp}: {ans}")

# You'll see: temp=0 gives same answer every time,
#             temp=1.5 gives wild, unpredictable names.


# ─────────────────────────────────────────────────────────────────
# PUTTING IT TOGETHER: Our Study Buddy so far
# ─────────────────────────────────────────────────────────────────
def ask_study_buddy(question: str) -> str:
    """
    Step 1 version: A simple Q&A function.
    Takes a question, returns a tutor-style answer.
    """
    messages = [
        SystemMessage(content=(
            "You are an expert Gen AI tutor named Buddy. "
            "Give clear, concise answers with examples. "
            "Keep answers under 150 words."
        )),
        HumanMessage(content=question),
    ]
    return llm.invoke(messages).content


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("YOUR STUDY BUDDY (Step 1)")
    print("=" * 60)
    print(ask_study_buddy("What is the difference between ML and Deep Learning?"))

"""
WHAT YOU LEARNED:
  ✅ LLMs predict next tokens — they don't "know" things like a database
  ✅ .invoke() → full response | .stream() → word by word
  ✅ SystemMessage controls persona, HumanMessage is the question
  ✅ Temperature 0 = consistent, higher = more creative
  ✅ LangChain abstracts away which provider you're using

NEXT → step_02_prompts.py
  Problem: Our system prompt is just a hardcoded string.
  What if we want to dynamically change the topic, difficulty, or format?
  → Prompt Templates solve this.
"""

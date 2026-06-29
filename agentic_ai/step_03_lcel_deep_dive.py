"""
03 - LCEL Chains: LangChain Expression Language
================================================

CONCEPTS COVERED:
  - The | pipe operator (LCEL)
  - RunnablePassthrough, RunnableLambda, RunnableParallel
  - Branching chains with RunnableBranch
  - Sequential chains (output of one → input of next)
  - itemgetter for dict manipulation
  - .bind() for locking in LLM parameters

WHAT IS LCEL?
  LangChain Expression Language is a declarative way to compose chains.
  Every component (prompt, LLM, parser, function) is a "Runnable" that
  can be piped together with |, just like Unix pipes.

  The key interface every Runnable exposes:
    .invoke(input)     → single call
    .stream(input)     → streaming
    .batch([inputs])   → parallel batch
    .with_fallbacks([...])  → fallback chain if this one fails
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)
parser = StrOutputParser()


# ── 1. The Pipe operator (|) — basics ────────────────────────────────────────
print("=" * 60)
print("1. Basic LCEL chain: prompt | llm | parser")
print("=" * 60)

chain = (
    ChatPromptTemplate.from_messages([("human", "What is {topic}?")])
    | llm
    | parser
)

# Every LCEL chain also exposes .invoke(), .stream(), .batch()
result = chain.invoke({"topic": "quantum computing"})
print(result[:200] + "...")


# ── 2. RunnablePassthrough — pass input through unchanged ─────────────────────
print("\n" + "=" * 60)
print("2. RunnablePassthrough — pass input through")
print("=" * 60)

from langchain_core.runnables import RunnablePassthrough

# Use case: you want the chain to also return the original input alongside output
chain_with_input = (
    RunnablePassthrough.assign(                    # add 'answer' key to the dict
        answer=ChatPromptTemplate.from_messages([("human", "{question}")]) | llm | parser
    )
)

result = chain_with_input.invoke({"question": "What is 5 * 7?"})
print(result)  # {'question': 'What is 5 * 7?', 'answer': '35'}


# ── 3. RunnableLambda — wrap any Python function as a Runnable ────────────────
print("\n" + "=" * 60)
print("3. RunnableLambda — custom Python function in a chain")
print("=" * 60)

from langchain_core.runnables import RunnableLambda

def add_exclamations(text: str) -> str:
    return text.upper() + " !!!"

chain = (
    ChatPromptTemplate.from_messages([("human", "In one word: {topic}")])
    | llm
    | parser
    | RunnableLambda(add_exclamations)   # ← custom function
)

result = chain.invoke({"topic": "fastest animal"})
print(result)   # CHEETAH !!!


# ── 4. RunnableParallel — run multiple chains simultaneously ─────────────────
print("\n" + "=" * 60)
print("4. RunnableParallel — fan-out, run chains in parallel")
print("=" * 60)

from langchain_core.runnables import RunnableParallel

joke_prompt = ChatPromptTemplate.from_messages([("human", "Tell me a short joke about {topic}")])
fact_prompt = ChatPromptTemplate.from_messages([("human", "Give me an interesting fact about {topic}")])

parallel_chain = RunnableParallel(
    joke=(joke_prompt | llm | parser),
    fact=(fact_prompt | llm | parser),
)

result = parallel_chain.invoke({"topic": "Python programming"})
print("JOKE:", result["joke"])
print("\nFACT:", result["fact"])


# ── 5. Sequential chain — chain output into next chain ───────────────────────
print("\n" + "=" * 60)
print("5. Sequential chain - output of chain 1 -> input of chain 2")
print("=" * 60)

# Chain 1: Generate a blog title
title_chain = (
    ChatPromptTemplate.from_messages([("human", "Generate a catchy blog title about {topic}")])
    | llm
    | parser
)

# Chain 2: Write an intro for a given title
intro_chain = (
    ChatPromptTemplate.from_messages([("human", "Write a 3-sentence intro for this blog title: {title}")])
    | llm
    | parser
)

# Compose: first get title, then use it for intro
full_chain = (
    title_chain
    | RunnableLambda(lambda title: {"title": title})   # reshape output → input
    | intro_chain
)

result = full_chain.invoke({"topic": "AI Agents"})
print(result)


# ── 6. RunnableBranch — conditional routing ───────────────────────────────────
print("\n" + "=" * 60)
print("6. RunnableBranch — route input to different chains")
print("=" * 60)

from langchain_core.runnables import RunnableBranch

# Classify the question first
classify_prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify the user question into exactly one word: 'math', 'history', or 'general'."),
    ("human", "{question}"),
])
classify_chain = classify_prompt | llm | parser

# Specialized chains
math_chain = (
    ChatPromptTemplate.from_messages([("system", "You are a math tutor."), ("human", "{question}")])
    | llm | parser
)
history_chain = (
    ChatPromptTemplate.from_messages([("system", "You are a history professor."), ("human", "{question}")])
    | llm | parser
)
general_chain = (
    ChatPromptTemplate.from_messages([("human", "{question}")])
    | llm | parser
)

router = RunnableBranch(
    (lambda x: "math"    in x["topic"].lower(), math_chain),
    (lambda x: "history" in x["topic"].lower(), history_chain),
    general_chain,  # default
)

full_router = (
    RunnablePassthrough.assign(topic=classify_chain)
    | router
)

questions = [
    {"question": "What is the integral of x^2?"},
    {"question": "Who was the first emperor of Rome?"},
    {"question": "What's a good recipe for pasta?"},
]
for q in questions:
    result = full_router.invoke(q)
    print(f"\nQ: {q['question']}")
    print(f"A: {result[:100]}...")


"""
KEY TAKEAWAYS:
  ✅ | is the LCEL pipe — composes Runnables left to right
  ✅ RunnablePassthrough: pass input through (or merge new keys with .assign())
  ✅ RunnableLambda: wrap any Python function into the chain
  ✅ RunnableParallel: fan-out and run branches concurrently
  ✅ RunnableBranch: conditional routing like if/elif/else
  ✅ Every chain is also a Runnable → chains can compose into bigger chains

NEXT: 04_memory/ — Adding memory to conversations
"""

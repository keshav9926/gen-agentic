# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║       AI STUDY BUDDY — STEP 2: Talking to it Properly           ║
║                    (Prompt Engineering)                         ║
╚══════════════════════════════════════════════════════════════════╝

WHAT WE'RE BUILDING TODAY:
  Upgrade from a fixed system prompt → dynamic, reusable templates.
  Add few-shot examples and Chain-of-Thought reasoning.

PROBLEM WITH STEP 1:
  Our prompt was hardcoded:
    "You are an expert Gen AI tutor..."

  What if we want:
    - Different difficulty levels (beginner / expert)?
    - Different output formats (bullet points / paragraph / JSON)?
    - Different topics with pre-filled context?

  Hardcoded strings don't scale. Prompt Templates do.

CONCEPT: Prompt Templates
──────────────────────────
  A template is a reusable prompt with {placeholders}.
  Think of it like an f-string, but smarter:

    "Explain {topic} at a {level} level using a {analogy_type} analogy."

  You define the structure once, fill in variables at runtime.
  This is the foundation of every LLM application.

CONCEPT: Few-Shot Prompting
────────────────────────────
  The LLM's behavior changes dramatically based on examples you give it.

  Without examples (zero-shot):
    Prompt: "Classify: 'I hate Mondays' → "
    Output: "Negative" (or a paragraph explaining it)

  With examples (few-shot):
    Prompt: "Classify:
             'I love pizza' → Positive
             'Traffic is awful' → Negative
             'I hate Mondays' → "
    Output: "Negative"  ← forced into your exact format

  Few-shot is the cheapest way to control output format + behavior.

CONCEPT: Chain-of-Thought (CoT)
─────────────────────────────────
  LLMs are better at hard problems when you ask them to "think aloud."

  Without CoT:
    Q: "If a train leaves at 9am at 60mph and another leaves at 10am at 80mph
        from 200 miles apart, when do they meet?"
    A: (often wrong)

  With CoT (just add "Think step by step"):
    A: "First, let me set up the equations... distance = rate × time...
        solving: they meet at 11:40am"  ← correct!

  WHY? Because generating intermediate steps forces the model to
  "commit" to a reasoning path before answering — reduces hallucination.
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from typing import Literal

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


# ─────────────────────────────────────────────────────────────────
# LESSON 2A: ChatPromptTemplate — Dynamic prompts
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("LESSON 2A: Prompt Templates")
print("=" * 60)

# Define a reusable template with {variables}
tutor_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert {subject} tutor. "
        "Explain concepts at a {level} level. "
        "Always end with one practice question."
    )),
    ("human", "Explain: {topic}"),
])

# Fill in the variables at runtime
filled_prompt = tutor_prompt.invoke({
    "subject": "Generative AI",
    "level": "beginner",
    "topic": "what is an attention mechanism",
})

print("Filled prompt messages:")
for msg in filled_prompt.messages:
    print(f"  [{msg.type}]: {msg.content[:80]}...")

response = llm.invoke(filled_prompt)
print("\nAnswer:")
print(response.content)


# ─────────────────────────────────────────────────────────────────
# LESSON 2B: Chaining with | (LCEL — LangChain Expression Language)
# ─────────────────────────────────────────────────────────────────
# Instead of: response = llm.invoke(prompt.invoke(vars))
# We write:   chain = prompt | llm | parser
#
# The | (pipe) operator chains callables together.
# Data flows LEFT to RIGHT:
#   dict → [prompt] → ChatPromptValue → [llm] → AIMessage → [parser] → str
#
# Benefits:
#   - Clean, readable pipeline
#   - Easy to swap components (change llm, change parser)
#   - Supports .stream(), .batch(), .ainvoke() automatically
print("\n" + "=" * 60)
print("LESSON 2B: LCEL Chains (prompt | llm | parser)")
print("=" * 60)

parser = StrOutputParser()  # Extracts .content from AIMessage → plain string

# Build the chain once
study_chain = tutor_prompt | llm | parser

# Invoke with different variables — same chain, different inputs
result = study_chain.invoke({
    "subject": "Generative AI",
    "level": "expert",
    "topic": "difference between LoRA and full fine-tuning",
})
print(result)


# ─────────────────────────────────────────────────────────────────
# LESSON 2C: Few-Shot Prompting — Teaching by Example
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("LESSON 2C: Few-Shot Prompting")
print("=" * 60)

# Examples teach the model the EXACT output format we want
examples = [
    {
        "question": "What is an LLM?",
        "answer": (
            "DEFINITION: A Large Language Model trained on massive text data.\n"
            "ANALOGY: Like a very well-read person who predicts what word comes next.\n"
            "KEY FACT: GPT-4 has ~1.8 trillion parameters."
        )
    },
    {
        "question": "What is a vector database?",
        "answer": (
            "DEFINITION: A database that stores and searches high-dimensional vectors.\n"
            "ANALOGY: Like a library that finds books by meaning, not just keywords.\n"
            "KEY FACT: ChromaDB, Pinecone, and FAISS are popular options."
        )
    },
]

# Template for each example pair
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{question}"),
    ("ai", "{answer}"),
])

# Combine examples into a FewShot prompt
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# Full prompt = system + few-shot examples + new question
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Gen AI tutor. Answer in the exact format shown."),
    few_shot_prompt,  # inject the examples here
    ("human", "{question}"),
])

chain = final_prompt | llm | parser
result = chain.invoke({"question": "What is RAG?"})
print(result)
# Notice: the output follows DEFINITION / ANALOGY / KEY FACT format
# because the model learned from the examples!


# ─────────────────────────────────────────────────────────────────
# LESSON 2D: Chain-of-Thought + Structured Output
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("LESSON 2D: Chain-of-Thought + JSON Output")
print("=" * 60)

# Define the exact structure we want back using Pydantic
class ConceptExplanation(BaseModel):
    concept: str = Field(description="The concept being explained")
    simple_explanation: str = Field(description="One sentence, no jargon")
    analogy: str = Field(description="A real-world analogy")
    example_code: str = Field(description="A short code snippet if applicable, else 'N/A'")
    difficulty: Literal["beginner", "intermediate", "advanced"]
    key_interview_question: str = Field(description="One likely interview question on this topic")

# JsonOutputParser forces the LLM to return valid JSON matching our schema
json_parser = JsonOutputParser(pydantic_object=ConceptExplanation)

structured_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a Gen AI expert. Think step by step, then return a JSON "
        "explanation of the concept. {format_instructions}"
    )),
    ("human", "Explain the concept: {concept}"),
]).partial(format_instructions=json_parser.get_format_instructions())

structured_chain = structured_prompt | llm | json_parser

result = structured_chain.invoke({"concept": "embedding vectors"})
print(f"Concept:    {result['concept']}")
print(f"Simple:     {result['simple_explanation']}")
print(f"Analogy:    {result['analogy']}")
print(f"Difficulty: {result['difficulty']}")
print(f"Interview:  {result['key_interview_question']}")


# ─────────────────────────────────────────────────────────────────
# PUTTING IT TOGETHER: Study Buddy v2 — Smart Q&A
# ─────────────────────────────────────────────────────────────────
def ask_study_buddy_v2(topic: str, level: str = "beginner") -> dict:
    """
    Step 2 version: Returns a structured explanation with CoT.
    Now we get consistent JSON output we can use in a UI.
    """
    return structured_chain.invoke({"concept": topic})


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("STUDY BUDDY v2 — Structured Output")
    print("=" * 60)
    result = ask_study_buddy_v2("attention mechanism in transformers")
    for key, val in result.items():
        print(f"\n{key.upper()}:\n  {val}")


"""
WHAT YOU LEARNED:
  ✅ ChatPromptTemplate → reusable, dynamic prompts with {variables}
  ✅ LCEL (|) → clean pipelines: prompt | llm | parser
  ✅ Few-shot prompting → show examples, control output format
  ✅ Chain-of-Thought → "think step by step" improves complex reasoning
  ✅ JsonOutputParser + Pydantic → force structured, type-safe output

  INTERVIEW INSIGHT:
    "How do you ensure an LLM always returns valid JSON?"
    → Use structured output / function calling + Pydantic validation.
    → Add retry logic if parsing fails.

NEXT → step_03_memory.py
  PROBLEM: Every time you call ask_study_buddy_v2(), it forgets
  the previous conversation. Ask "Who am I?" → it won't know.
  → Memory solves this.
"""

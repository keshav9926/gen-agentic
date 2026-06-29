# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║           AGENTIC AI — STEP 5: Corrective RAG (CRAG)             ║
║                      (Self-Grading & Web Fallback)               ║
╚══════════════════════════════════════════════════════════════════╝

WHAT WE'RE LEARNING TODAY:
  - Designing a Corrective RAG (CRAG) self-corrective pipeline.
  - Creating a Document Grader node to evaluate retrieval relevance.
  - Making conditional routing decisions based on retrieval score quality.
  - Implementing fallback search (web search) if internal documents are irrelevant.
  - Synthesizing validated documents into the final answer.

CONCEPT: What is Corrective RAG?
────────────────────────────────
  Naive RAG grabs the top-k chunks from a vector DB and sends them directly to
  the LLM, even if they are irrelevant. This can lead to hallucination.
  
  Corrective RAG (CRAG) adds a self-correcting loop:
    1. Retrieve chunks from the database.
    2. Grade each chunk (using a fast, focused LLM prompt).
    3. Route:
       - If chunks are highly relevant → proceed to Generation.
       - If chunks are irrelevant or missing → trigger Web Search fallback to
         gather fresh context.
       - Synthesize context from the best sources to answer.
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

# Load env variables
load_dotenv()

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ─────────────────────────────────────────────────────────────────
# LESSON 1: Define Chunks and Grader
# ─────────────────────────────────────────────────────────────────

# In-memory document storage
DOCUMENTS_DB = {
    "transformers": "The Transformer model uses self-attention mechanisms to process tokens in parallel.",
    "lora": "LoRA (Low-Rank Adaptation) freezes pre-trained weights and injects trainable adapter matrices.",
    "dpo": "Direct Preference Optimization simplifies alignment by optimizing policy directly on human preference data."
}

class CragState(TypedDict):
    query: str
    documents: List[str]      # list of document texts retrieved/validated
    web_search_needed: bool   # fallback trigger flag
    generation: str           # final answer

# LLM Document Grader: evaluates if a document is relevant to the query
def grade_document(query: str, doc: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a grader checking document relevance. "
                   "Respond with exactly one word: 'yes' if the document is relevant to the query, or 'no' if it is not."),
        ("human", "Query: {query}\nDocument: {doc}")
    ])
    grader_chain = prompt | llm | StrOutputParser()
    result = grader_chain.invoke({"query": query, "doc": doc}).strip().lower()
    return "yes" if "yes" in result else "no"


# ─────────────────────────────────────────────────────────────────
# LESSON 2: Define Graph Nodes
# ─────────────────────────────────────────────────────────────────

def retrieve_node(state: CragState) -> dict:
    print("\n[Node: Retrieve]")
    query = state["query"].lower()
    retrieved = []
    
    # Simple keyword match retriever simulation
    for key, text in DOCUMENTS_DB.items():
        if key in query:
            retrieved.append(text)
            
    print(f"   Retrieved {len(retrieved)} document(s) from database.")
    return {"documents": retrieved, "web_search_needed": False}


def grade_documents_node(state: CragState) -> dict:
    print("\n[Node: Grade Documents]")
    query = state["query"]
    docs = state["documents"]
    
    valid_docs = []
    search_needed = False
    
    if not docs:
        print("   No documents retrieved. Web search fallback will be triggered.")
        search_needed = True
    else:
        for doc in docs:
            grade = grade_document(query, doc)
            if grade == "yes":
                print(f"   Document graded: RELEVANT -> Keeping context.")
                valid_docs.append(doc)
            else:
                print(f"   Document graded: IRRELEVANT -> Filtering out.")
                
        # If all retrieved docs were filtered out, fallback is needed
        if not valid_docs:
            print("   No relevant documents remaining. Web search fallback needed.")
            search_needed = True
            
    return {"documents": valid_docs, "web_search_needed": search_needed}


def web_search_node(state: CragState) -> dict:
    print("\n[Node: Web Search Fallback]")
    query = state["query"]
    
    # Simulate a web search integration
    print(f"   Searching the web for '{query}'...")
    simulated_web_result = (
        f"Web Search Excerpt: Recent research shows that '{query}' is a key technique "
        f"used to improve LLM accuracy, alignment, or optimization dynamically at scale."
    )
    
    # Append the web search result to our active documents context
    updated_docs = state["documents"] + [simulated_web_result]
    return {"documents": updated_docs, "web_search_needed": False}


def generate_node(state: CragState) -> dict:
    print("\n[Node: Generate Answer]")
    query = state["query"]
    context = "\n".join(state["documents"])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert tutor. Answer the user query using ONLY the provided context. "
                   "If you do not know or the context is empty, say 'No relevant information found.'"),
        ("human", "Context:\n{context}\n\nQuery: {query}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "query": query})
    return {"generation": answer}


# ─────────────────────────────────────────────────────────────────
# LESSON 3: Define Routing Logic and Compile
# ─────────────────────────────────────────────────────────────────

# Conditional edge: decides whether to route to web_search or directly to generate
def decide_routing(state: CragState) -> str:
    if state["web_search_needed"]:
        return "web_search"
    return "generate"

# Build CRAG Graph
builder = StateGraph(CragState)

builder.add_node("retrieve", retrieve_node)
builder.add_node("grade", grade_documents_node)
builder.add_node("web_search", web_search_node)
builder.add_node("generate", generate_node)

builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "grade")

# Add routing edge from grade node
builder.add_conditional_edges(
    "grade",
    decide_routing,
    {
        "web_search": "web_search",
        "generate": "generate"
    }
)
builder.add_edge("web_search", "generate")
builder.add_edge("generate", END)

crag_agent = builder.compile()


# ─────────────────────────────────────────────────────────────────
# LESSON 4: Running Test Scenarios
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test Scenario 1: Query hits internal database (no web fallback)
    print("=" * 60)
    print("SCENARIO 1: INTERNAL RETRIEVAL MATCH (LoRA)")
    print("=" * 60)
    res_1 = crag_agent.invoke({"query": "Explain how LoRA works."})
    print(f"\nFinal Generation:\n{res_1['generation']}")
    
    # Test Scenario 2: Query misses database (triggers web fallback)
    print("\n" + "=" * 60)
    print("SCENARIO 2: IRRELEVANT RETRIEVAL -> WEB FALLBACK (reinforcement learning)")
    print("=" * 60)
    res_2 = crag_agent.invoke({"query": "What is PPO in reinforcement learning?"})
    print(f"\nFinal Generation:\n{res_2['generation']}")


"""
WHAT YOU LEARNED:
  ✅ Naive RAG is prone to hallucination; CRAG adds self-corrective checkpoints.
  ✅ Grader LLM checks retrieval relevance and filters out noise.
  ✅ Conditional routing handles fallbacks (e.g. web search) dynamically based
     on graph state properties.

INTERVIEW INSIGHTS:
  "What is Corrective RAG (CRAG) and how does it differ from Naive RAG?"
  -> Naive RAG passes retrieved documents directly to the generator. CRAG evaluates
     the retrieved documents first, discards irrelevant documents, and performs
     web search as a fallback if the local retrieval quality is too low.

  "How does Self-RAG differ from CRAG?"
  -> Self-RAG uses self-reflection tokens generated by a single fine-tuned model
     to evaluate its own retrieval and output quality. CRAG uses a pipeline/graph
     approach where grading and searching are distinct modular nodes.
"""

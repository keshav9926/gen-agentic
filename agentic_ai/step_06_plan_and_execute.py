# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║         AGENTIC AI — STEP 6: Plan-and-Execute Agent Pattern      ║
╚══════════════════════════════════════════════════════════════════╝

WHAT WE'RE LEARNING TODAY:
  - The Plan-and-Execute agent architecture for complex tasks.
  - Designing a Planner node to decompose queries into sub-tasks.
  - Designing an Executor node to process tasks sequentially.
  - Implementing a Re-planner node to adjust plans dynamically based on outcomes.
  - Mitigating agent drift on long-horizon, multi-step goals.

CONCEPT: Plan-and-Execute Agent Pattern
───────────────────────────────────────
  Standard ReAct agents decide their next step one action at a time. For complex,
  long-horizon goals (e.g. writing a paper with citation research and coding),
  ReAct agents suffer from "drift" — they forget the ultimate goal or get stuck looping.
  
  The Plan-and-Execute pattern separates planning from execution:
    1. PLANNER: Takes the user query and generates a structured plan (list of steps).
    2. EXECUTOR: Takes the current step and executes it using tools.
    3. RE-PLANNER: Looks at the executor result, updates the list of tasks (adds, deletes,
       or marks steps as done), and either routes back to execution or returns the final result.
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, List
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langgraph.graph import StateGraph, END

# Load env variables
load_dotenv()

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ─────────────────────────────────────────────────────────────────
# LESSON 1: Define Schemas and Planner
# ─────────────────────────────────────────────────────────────────

# Pydantic schemas to structure planning outputs
class Plan(BaseModel):
    steps: List[str] = Field(description="List of sequential steps/tasks to complete the goal.")

class ReplannerOutput(BaseModel):
    is_completed: bool = Field(description="True if the plan is fully done and goal is reached.")
    next_step: str = Field(description="The next task to execute. Empty if is_completed is True.")
    updated_plan: List[str] = Field(description="The remaining, adjusted list of steps.")

# State representation
class PlanState(TypedDict):
    input_query: str
    plan: List[str]
    current_step: str
    step_results: List[str] # logs of finished step results
    final_answer: str

# ── Planner Node ──────────────────────────────────────────────────
# Takes the user query and breaks it down into a list of tasks
def planner_node(state: PlanState) -> dict:
    print("\n[Node: Planner]")
    query = state["input_query"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert planner. Break down the user's complex goal into a clear, "
                   "sequential list of 3 concrete sub-tasks."),
        ("human", "Goal: {query}")
    ])
    
    planner_llm = llm.with_structured_output(Plan)
    planner_chain = prompt | planner_llm
    
    plan_output = planner_chain.invoke({"query": query})
    print(f"   Generated Plan: {plan_output.steps}")
    
    return {
        "plan": plan_output.steps,
        "current_step": plan_output.steps[0]
    }


# ─────────────────────────────────────────────────────────────────
# LESSON 2: Define Executor and Replanner
# ─────────────────────────────────────────────────────────────────

# ── Executor Node ─────────────────────────────────────────────────
# Simulates executing the current task using tools
def executor_node(state: PlanState) -> dict:
    print(f"\n[Node: Executor] Processing step: '{state['current_step']}'")
    step = state["current_step"]
    
    # Simple simulated execution based on step description
    execution_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an agent executing a single task. Provide a brief, factual 1-sentence outcome."),
        ("human", "Execute this task: {step}")
    ])
    execution_chain = execution_prompt | llm | StrOutputParser()
    result = execution_chain.invoke({"step": step})
    
    print(f"   Result: {result}")
    return {"step_results": state["step_results"] + [f"Task: {step} -> Result: {result}"]}


# ── Replanner Node ────────────────────────────────────────────────
# Reviews outcomes, marks task done, updates plan, and checks for completion
def replanner_node(state: PlanState) -> dict:
    print("\n[Node: Replanner]")
    query = state["input_query"]
    plan = state["plan"]
    results = state["step_results"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a manager reviewing progress. Decide if the user's goal is complete.\n"
                   "Original Goal: {query}\n"
                   "Plan: {plan}\n"
                   "Completed Tasks Results:\n{results}\n\n"
                   "Provide: (1) is_completed: true or false. "
                   "(2) next_step: if not completed, specify the next task. "
                   "(3) updated_plan: list of remaining steps including any adjustments."),
        ("human", "Evaluate current progress and update the plan.")
    ])
    
    replanner_llm = llm.with_structured_output(ReplannerOutput)
    replanner_chain = prompt | replanner_llm
    
    output = replanner_chain.invoke({"query": query, "plan": plan, "results": "\n".join(results)})
    
    if output.is_completed:
        print("   Goal reached! Compiling final response.")
        # Generate final output synthesizing all results
        summary_prompt = f"Summarize the final outcomes for the user's original goal: '{query}' based on these logs:\n" + "\n".join(results)
        summary = llm.invoke(summary_prompt).content
        return {"final_answer": summary, "plan": []}
    else:
        print(f"   Goal NOT yet reached. Next step: '{output.next_step}'")
        print(f"   Updated Plan: {output.updated_plan}")
        return {
            "plan": output.updated_plan,
            "current_step": output.next_step
        }


# ─────────────────────────────────────────────────────────────────
# LESSON 3: Build the Graph and Run
# ─────────────────────────────────────────────────────────────────

def should_continue(state: PlanState) -> str:
    # If the plan is empty, it means we finished
    if not state["plan"]:
        return "end"
    return "execute"

# Build Graph
builder = StateGraph(PlanState)

builder.add_node("planner", planner_node)
builder.add_node("executor", executor_node)
builder.add_node("replanner", replanner_node)

builder.set_entry_point("planner")
builder.add_edge("planner", "executor")
builder.add_edge("executor", "replanner")

# Route back to executor or end based on remaining plan steps
builder.add_conditional_edges(
    "replanner",
    should_continue,
    {
        "execute": "executor",
        "end": END
    }
)

plan_agent = builder.compile()

if __name__ == "__main__":
    query = "Draft a short comparative outline between transformers and convolutional networks for image classification."
    
    # Initialize state
    initial_state = {
        "input_query": query,
        "plan": [],
        "current_step": "",
        "step_results": [],
        "final_answer": ""
    }
    
    result = plan_agent.invoke(initial_state)
    
    print("\n" + "=" * 60)
    print("FINAL OUTCOME:")
    print("=" * 60)
    print(result["final_answer"])


"""
WHAT YOU LEARNED:
  ✅ Plan-and-Execute architecture splits task decomposition from execution.
  ✅ Planner node generates a structured Pydantic task list.
  ✅ Replanner node checks progress and updates the remaining plan steps.
  ✅ This pattern prevents agentic drift on long-horizon, multi-step goals.

INTERVIEW INSIGHTS:
  "When would you prefer a Plan-and-Execute agent over a ReAct agent?"
  -> For complex, long-horizon tasks that require multiple steps (e.g. software development
     or data analysis), ReAct agents tend to get distracted or enter infinite loops.
     Plan-and-Execute maintains a global plan, preventing drift and improving reliability.

  "How does a Replanner handle execution failures?"
  -> If a step fails, the replanner reads the error in the results and modifies
     the plan dynamically (e.g., adding a debugging step or rewriting the failed task).
"""

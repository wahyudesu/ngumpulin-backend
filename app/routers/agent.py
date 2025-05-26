import os
import getpass
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from typing_extensions import TypedDict

# --- Konfigurasi API Key Groq (bisa lewat environment variable) ---
load_dotenv()

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

# _set_env("GROQ_API_KEY")  # uncomment jika perlu input manual

# Inisialisasi LLM dari Groq (contoh: gemma2-9b-it)
llm = ChatGroq(model="gemma2-9b-it", api_key=os.getenv("GROQ_API_KEY"))

# --- Tipe Data State ---
class AssignmentMeta(TypedDict):
    title: str
    description: str

class State(TypedDict):
    assignment_meta: AssignmentMeta
    assignment_content: str
    persona: str
    summary: str
    relevance_analysis: str
    feedback_analysis: str
    personalized_feedback: str
    combined_output: str

# --- Node Agent Functions ---
def input_meta(state: State) -> dict:
    return {"assignment_meta": state["assignment_meta"]}

def input_content(state: State) -> dict:
    return {"assignment_content": state["assignment_content"]}

def summarizer_agent(state: State) -> dict:
    content = state["assignment_content"]
    msg = llm.invoke(f"Summarize the following assignment content:\n\n{content}")
    return {"summary": msg.content}

def relevance_agent(state: State) -> dict:
    title = state["assignment_meta"]["title"]
    desc = state["assignment_meta"]["description"]
    prompt = f"""Analyze the relevance between the following title and description of an assignment:

Title: {title}
Description: {desc}

Does the title appropriately reflect the content described? Provide analysis."""
    msg = llm.invoke(prompt)
    return {"relevance_analysis": msg.content}

def aggregator(state: State) -> dict:
    summary = state["summary"]
    relevance = state["relevance_analysis"]
    persona = state["persona"]

    feedback_prompt = f"""You are an academic evaluator. Provide constructive feedback based on the following:

SUMMARY:
{summary}

RELEVANCE ANALYSIS:
{relevance}"""
    feedback = llm.invoke(feedback_prompt).content

    personalization_prompt = f"""Personalize the following academic feedback based on the following style or instruction:

INSTRUCTION:
{persona}

FEEDBACK:
{feedback}"""
    personalized = llm.invoke(personalization_prompt).content

    combined = f"🎓 Final Personalized Feedback:\n\n{personalized}"
    return {
        "feedback_analysis": feedback,
        "personalized_feedback": personalized,
        "combined_output": combined
    }

# --- Bangun Workflow ---
builder = StateGraph(State)
builder.add_node("input_meta", input_meta)
builder.add_node("input_content", input_content)
builder.add_node("relevance_agent", relevance_agent)
builder.add_node("summarizer_agent", summarizer_agent)
builder.add_node("aggregator", aggregator)

builder.add_edge(START, "input_meta")
builder.add_edge(START, "input_content")
builder.add_edge("input_meta", "relevance_agent")
builder.add_edge("input_content", "summarizer_agent")
builder.add_edge("relevance_agent", "aggregator")
builder.add_edge("summarizer_agent", "aggregator")
builder.add_edge("aggregator", END)

workflow = builder.compile()

# --- FastAPI Setup ---
app = FastAPI(title="AI Assignment Feedback Agent")

class AssignmentRequest(BaseModel):
    title: str
    description: str
    content: str
    persona: str

class AssignmentResponse(BaseModel):
    summary: str
    relevance_analysis: str
    feedback_analysis: str
    personalized_feedback: str
    combined_output: str

@app.post("/feedback", response_model=AssignmentResponse)
def generate_feedback(payload: AssignmentRequest):
    initial_state: State = {
        "assignment_meta": {
            "title": payload.title,
            "description": payload.description,
        },
        "assignment_content": payload.content,
        "persona": payload.persona,
        "summary": "",
        "relevance_analysis": "",
        "feedback_analysis": "",
        "personalized_feedback": "",
        "combined_output": ""
    }
    result = workflow.invoke(initial_state)
    return result
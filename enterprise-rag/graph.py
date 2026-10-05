from typing import Literal
from typing_extensions import TypedDict
from routing import classify_query
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage
from rag import rag_pipeline

class GraphState(TypedDict, total=False):
    query: str
    history : list[BaseMessage]
    session_id : str
    route: Literal["greeting", "rag", "out_of_scope"]
    answer: str
    sources: list


def router_node(state: GraphState):
    route = classify_query(state["query"])
    return {
        "route": route
    }

def greeting_node(state: GraphState):
    return {
        "answer": "Hey! 👋 How can I help you today?",
        "sources": []
    }

def out_of_scope_node(state: GraphState):
    return {
        "answer": (
            "I'm designed to answer questions based on the "
            "enterprise knowledge base."
        ),
        "sources": []
    }

def rag_node(state: GraphState):
    rag_result = rag_pipeline(
        query = state["query"],
        history = state["history"],
        session_id = state["session_id"]
    )
    return {
        "answer": rag_result["answer"],
        "sources": rag_result["sources"]
    }

def route_from_classifier(state: GraphState):
    return state["route"]


builder = StateGraph(GraphState)
builder.add_node("router", router_node)
builder.add_node("greeting", greeting_node)
builder.add_node("rag", rag_node)
builder.add_node("out_of_scope", out_of_scope_node)

builder.add_edge(START, "router")


builder.add_conditional_edges(
    "router",
    route_from_classifier,
    {
        "greeting": "greeting",
        "rag": "rag",
        "out_of_scope": "out_of_scope",
    },
)

builder.add_edge("greeting", END)
builder.add_edge("rag", END)
builder.add_edge("out_of_scope", END)

graph = builder.compile()



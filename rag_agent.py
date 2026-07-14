from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain.messages import AnyMessage
from tools import rag_model_with_structured_output
from message import rag_system_msg

class AgentState(TypedDict):
    messages: list[AnyMessage]
    search_results: list

rag_graph = StateGraph(AgentState)

#store the retrieved data in a cache for faster lookup  
def retrieve_data_node(state:AgentState):
    user_query = state["messages"][-1].content
    search_results = search(user_query)
    state["search_results"] = search_results
    return state

def answer_user_query_node(state:AgentState):
    # This node would typically involve calling an LLM to generate a response based on the retrieved data.
    user_query = state["messages"][-1].content
    search_results = state.get("search_results", [])
    response = rag_model_with_structured_output.invoke([rag_system_msg, *state["messages"], {"query": user_query, "retrieved_data": search_results}])
    state["messages"].append(response)
    return state

rag_graph = StateGraph(AgentState)

rag_graph.add_node("retrieve_data", retrieve_data_node)
rag_graph.add_node("answer_user_query", answer_user_query_node)

rag_graph.add_edge(START, "retrieve_data")
rag_graph.add_edge("retrieve_data", "answer_user_query")
rag_graph.add_edge("answer_user_query", END)

rag_agent = rag_graph.compile()
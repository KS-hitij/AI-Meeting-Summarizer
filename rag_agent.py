from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain.messages import AnyMessage, AIMessage, HumanMessage
from tools import search, SummarizingAnswer
from message import rag_system_msg, judge_system_message, improve_system_message
from models import (
    rag_model_with_structured_output,
    judge_model_with_structured_output,
    improve_model_with_structured_output,
)


class AgentState(TypedDict):
    messages: list[AnyMessage]
    project_id: str
    search_results: list
    llm_output: str


rag_graph = StateGraph(AgentState)


# retrieve the data from the vector database based on the user query and project id
def retrieve_data_node(state: AgentState):
    if not state.get("project_id"):
        raise ValueError("Project Id not passed to agent")

    project_id = state.get("project_id")
    user_query = state["messages"][-1]

    from redis_client import r

    cached_result = r.get(f"project:{project_id}:query:{user_query['content']}")
    if cached_result:
        state["search_results"] = cached_result
        return state
    import json

    search_results = search.invoke(
        {"query": user_query["content"], "project_id": project_id}
    )
    r.set(
        f"project:{project_id}:query:{user_query['content']}",
        json.dumps(search_results),
        ex=300,
    )
    state["search_results"] = search_results
    return state


# This node would typically involve calling an LLM to generate a response based on the retrieved data.
def answer_user_query_node(state: AgentState):
    user_query = state["messages"][-1]["content"]
    search_results = state.get("search_results", [])
    query_with_context = HumanMessage(
        content=f"Here is the user query: {user_query}\nHere is the data retrieved information from the vector database:\n{search_results}"
    )
    response = rag_model_with_structured_output.invoke(
        [rag_system_msg, query_with_context]
    )
    state["llm_output"] = response.response
    return state


def judge_llm_answer_node(state: AgentState):
    llm_response = state.get("llm_output")
    search_results = state.get("search_results", [])
    user_query = state["messages"][-1]["content"]
    judge_prompt = HumanMessage(
        content=f"Query: {user_query}\nAnswer: {llm_response}\nContext: {search_results} Judge only on the basis of the provided context"
    )
    response = judge_model_with_structured_output.invoke(
        input=[judge_system_message, judge_prompt]
    )
    if response.accurate:
        ai_output = AIMessage(content=llm_response)
        state["messages"].append(ai_output)
        return "true"
    else:
        return "false"


def improve_llm_response_node(state: AgentState):
    llm_response = state.get("llm_output")
    search_results = state.get("search_results", [])
    user_query = state["messages"][-1]["content"]
    prompt = HumanMessage(
        content=f"User Query: {user_query}\nPrevious Response:{llm_response}\nRetrieved Information:{search_results}"
    )
    response = improve_model_with_structured_output.invoke(
        input=[improve_system_message, prompt]
    )
    ai_output = AIMessage(content=response.response)
    state["messages"].append(ai_output)
    return state


rag_graph = StateGraph(AgentState)

rag_graph.add_node("retrieve_data", retrieve_data_node)
rag_graph.add_node("answer_user_query", answer_user_query_node)
rag_graph.add_node("improve_llm_response", improve_llm_response_node)

rag_graph.add_edge(START, "retrieve_data")
rag_graph.add_edge("retrieve_data", "answer_user_query")
rag_graph.add_conditional_edges(
    "answer_user_query",
    judge_llm_answer_node,
    {"true": END, "false": "improve_llm_response"},
)
rag_graph.add_edge("improve_llm_response", END)

rag_agent = rag_graph.compile()

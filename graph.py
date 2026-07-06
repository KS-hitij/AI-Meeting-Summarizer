from langgraph.graph import StateGraph
from typing import TypedDict,Optional
from langchain.messages import SystemMessage
from tools import model

class File(TypedDict):
    file_name:str
    file_path:str
    file_type:str

class AgentState(TypedDict):
    file: Optional[File]
    response:str
    query:str
    llm_calls:int


graph = StateGraph(AgentState)



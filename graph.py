from langgraph.graph import StateGraph, START, END
from typing import TypedDict,Optional
from langchain.messages import HumanMessage
from tools import transcribe, model_with_structured_output
from message import system_msg
import os

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


def decide_text_or_audio(state:AgentState):
    """Decides whether the file passed is an audio file or a text file and then decides the flow accordingly"""
    if not state.get('file'):
        raise ValueError("No file found in agent state. Expected file with `file_type`.")

    file_type = state['file'].get('file_type')
    if file_type == 'audio':
        return "transcribe"
    elif file_type == 'text':
        return "summarize"
    else:
        raise ValueError(
            f"Unsupported file type: {file_type!r}. "
            "Expected 'audio' or 'text'."
        )



def transcribe_node(state:AgentState)->AgentState:
    """Performs transcription of the audio file using the transcribe tool."""
    if not state.get('file'):
        raise ValueError("No file found in agent state. Expected an audio file entry.")

    file_path = state['file'].get('file_path')
    if not file_path:
        raise ValueError("Missing file_path in state['file'].")

    if state['file'].get('file_type') != 'audio':
        raise ValueError("transcribe_node only accepts audio files.")

    transcription = transcribe.invoke({"file_path":file_path})

    state['response'] = transcription
    return state

def summarize_node(state:AgentState):
    """Using the transcription provided from the user, generate a summary of the meeting"""
    if state["file"]["file_type"] == 'text':
        file_path = state['file']["file_path"]
        if not os.path.exists(file_path):
            return {"error": "File not found"}
        
        with open(file_path, "r") as file:
            content = file.read()
        transcription = content
    else:
        transcription = state['response']
    prompt = f"Here is the transcription of the meeting:\n{transcription}"
    response = model_with_structured_output.invoke([system_msg,HumanMessage(content=prompt)])
    state['response'] = response
    return state


graph.add_node("transcribe",transcribe_node)
graph.add_node("summarize",summarize_node)

graph.add_edge("transcribe","summarize")
graph.add_edge("summarize",END)
graph.add_conditional_edges(
    START,
    decide_text_or_audio,
    {
        "transcribe":"transcribe",
        "summarize":"summarize"
    }
)

agent = graph.compile()
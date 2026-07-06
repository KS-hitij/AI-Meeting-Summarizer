from langgraph.graph import StateGraph
from typing import TypedDict,Optional
from langchain.messages import SystemMessage
from tools import model,transcribe

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



def transcribe_node(state:AgentState):
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


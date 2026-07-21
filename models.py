import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

model = init_chat_model(
    "google_genai:gemini-3.1-flash-lite",
)


class ActionItem(BaseModel):
    owner: str
    task: str


class OpenQuestion(BaseModel):
    question: str
    asked_by: str


class Risk(BaseModel):
    risk: str


class SummarizingAnswer(BaseModel):
    title: str
    summary: str
    action_items: list[ActionItem]
    open_questions: list[OpenQuestion]
    risks: Optional[Risk]


class JudgeAnswer(BaseModel):
    accurate: bool


class Rag_Answer(BaseModel):
    response: str


rag_model_with_structured_output = model.with_structured_output(Rag_Answer)
summarizing_model_with_structured_output = model.with_structured_output(
    SummarizingAnswer
)
judge_model_with_structured_output = model.with_structured_output(JudgeAnswer)
improve_model_with_structured_output = model.with_structured_output(Rag_Answer)

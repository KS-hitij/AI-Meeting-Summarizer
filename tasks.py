from agent import summarizer_agent
from message import system_msg
import os
from celery import Celery

celery = Celery(
    "tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1"
)

@celery.task(bind=True)
def summarize_file(self,tmp_path:str):
    result = summarizer_agent.invoke({
        "messages": [system_msg, {"role": "user", "content": f"Here is the file path: {tmp_path}"}]
    })
    try:
        return result["structured_response"]
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
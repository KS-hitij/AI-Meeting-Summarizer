from message import system_msg
import os
from celery import Celery
from graph import agent

CELERY_REDIS_CLIENT = os.getenv('CELERY_REDIS_CLIENT')

celery = Celery(
    "tasks",
    broker=CELERY_REDIS_CLIENT,
    backend=CELERY_REDIS_CLIENT
)

@celery.task(bind=True)
def summarize_file(self,tmp_path:str,file_name:str):
    result = agent.invoke({
        "file":{
            "file_name":file_name,
            "file_path":tmp_path,
            "file_type":"text"
        }
    })
    try:
        answer =  result["response"]
        return answer.model_dump()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
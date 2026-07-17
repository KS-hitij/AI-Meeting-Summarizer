import os
from celery import Celery
from rag_agent import rag_agent
from summarize_agent import summarize_agent
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CELERY_REDIS_CLIENT = os.getenv('CELERY_REDIS_CLIENT')

celery = Celery(
    "tasks",
    broker=CELERY_REDIS_CLIENT,
    backend=CELERY_REDIS_CLIENT
)

@celery.task(bind=True)
def summarize_file(self,tmp_path:str,file_name:str,file_type:str,project_id:str):
    result = summarize_agent.invoke({
        "file":{
            "file_name":file_name,
            "file_path":tmp_path,
            "file_type":file_type
        },
        "project_id":project_id
    })
    try:
        answer =  result["response"]
        return answer.model_dump()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@celery.task(bind=True)
def rag(self,query:str,project_id:str):
    result = rag_agent.invoke({
        "messages":[{
            "type":"human",
            "content":query
        }],
        "project_id":project_id
    })
    try:
        answer =  result["messages"][-1]
        return answer.model_dump()
    finally:
        pass
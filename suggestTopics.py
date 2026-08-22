# uvicorn suggestTopics:app --reload

import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import json
from fastapi import FastAPI
import mpire
import os
os.environ["OMP_NUM_THREADS"] = "1"
import time
import multiprocessing 
from mpire import WorkerPool
from pprint import pprint
from typing import List
from fastapi import FastAPI, Query
from typing import List

import asyncio
from fastapi import FastAPI, Query
from typing import List


load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key not found!")
client=Groq(api_key=my_api_key)

def get_choice(response):
    return response.choices[0].message.content

def get_response(model,messages,response_format):
    if response_format:
        return client.chat.completions.create(model=model, messages=messages, response_format=response_format)
    return client.chat.completions.create(model=model, messages=messages)
    

def run_topicCreation_execution(user_prompt:str,model:str)->str:
    role1 = "system"
    role2 = "user"
    ROLE = "role"
    CONTENT = "content"
    messages = [
        {ROLE:role1,CONTENT:f"Suggest only 1 topic in 2-3 words on the basis of this prompt."},
        {ROLE:role2,CONTENT:user_prompt},
    ]
    
    topic_response = get_response(model,messages,None)
    topic = get_choice(topic_response).strip().lower()
    return {"request":user_prompt,"response":topic}


from fastapi import FastAPI
from fastapi.responses import RedirectResponse

app = FastAPI(
    title="My API",
    version="1.0.0"
)

@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")
    


@app.get("/suggestTopics/")
def read_item(model:str,progress_bar:bool,user_prompts: List[str] = Query(...)):
    num_cores = max(min(multiprocessing.cpu_count() // 2, 2),1)
    results = []
    for user_prompt in user_prompts:
        my_dict = {
            "user_prompt":user_prompt,
            "model":model
        }
        results.append(my_dict)
    with WorkerPool(n_jobs=num_cores,daemon=False) as pool:
        results = pool.map(run_topicCreation_execution, results, progress_bar=progress_bar)
    
    return results


        












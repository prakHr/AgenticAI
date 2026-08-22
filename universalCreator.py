# uvicorn universalCreator:app --reload

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
    
def run_universalCreation_execution(topic:str,model:str,topic_type:str)->str:
    role1 = "system"
    role2 = "user"
    ROLE = "role"
    CONTENT = "content"
    messages = [
        {ROLE:role1,CONTENT:f"Create a {topic_type} on the basis of this topic."},
        {ROLE:role2,CONTENT:topic},
    ]
    
    story_response = get_response(model,messages,None)
    story = get_choice(story_response).strip().lower()
    return {"request":topic,"response":story}

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

app = FastAPI(
    title="My API",
    version="1.0.0"
)

@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

@app.get("/universalCreator/")
def read_item(model:str,user_prompt:str,topic_type:str):
    results = run_universalCreation_execution(user_prompt,model,topic_type)
    return results


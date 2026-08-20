# uvicorn planner:app --reload

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
        return client.chat.completions.create(model=model, messages=messages,response_format = response_format)
    return client.chat.completions.create(model=model, messages=messages)
    
def run_classification_execution(user_prompt:str,model:str,labels:list)->str:
    role1 = "system"
    role2 = "user"
    ROLE = "role"
    CONTENT = "content"
    # model="openai/gpt-oss-120b"
    messages = [
        {ROLE:role1,CONTENT:f"Classify the request into one of {labels}. Reply with just the label."},
        {ROLE:role2,CONTENT:user_prompt},
    ]
    
    route_response = get_response(model,messages,None)
    label = get_choice(route_response).strip().lower()
    return {"request":user_prompt,"response":label}


app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/classify/")
def read_item(model:str,user_prompt:str,labels: List[str] = Query(...)):
    labels.append("other")
    labels = list(set([l.lower() for l in labels]))
    results = run_classification_execution(user_prompt,model,labels)
    return results


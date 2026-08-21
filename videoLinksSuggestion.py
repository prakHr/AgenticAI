# uvicorn videoLinksSuggestion:app --reload

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
    
def run_videoLinksSuggestion_execution(user_prompt:str,model:str)->str:
    role1 = "system"
    role2 = "user"
    ROLE = "role"
    CONTENT = "content"
    messages = [
        {ROLE:role1,CONTENT:f"Suggest some youtube videos on the basis of this prompt."},
        {ROLE:role2,CONTENT:user_prompt},
    ]
    
    topic_response = get_response(model,messages,None)
    topic = get_choice(topic_response)
    return {"request":user_prompt,"response":topic}


app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/videoLinksSuggestion/")
def read_item(model:str,user_prompt:str):
    results = run_videoLinksSuggestion_execution(user_prompt,model)
    return results













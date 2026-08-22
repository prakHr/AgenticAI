# uvicorn songChoreographies:app --reload

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
    

def run_musicChoreography_execution(music_type:str,model:str)->str:
    role1 = "system"
    role2 = "user"
    ROLE = "role"
    CONTENT = "content"
    messages = [
        {ROLE:role1,CONTENT:f"Give a unique choreography on the basis of this music type. Only give choreography if it is a valid type of music. Please do not hallucinate."},
        {ROLE:role2,CONTENT:music_type},
    ]
    
    topic_response = get_response(model,messages,None)
    topic = get_choice(topic_response)
    return {"request":music_type,"response":topic}


app = FastAPI()


@app.get("/songChoreographies/")
def read_item(model:str,progress_bar:bool,music_types: List[str] = Query(...)):
    num_cores = max(min(multiprocessing.cpu_count() // 2, 2),1)
    results = [{"music_type" : music_type, "model" : model} for music_type in list(set(music_types))]
    with WorkerPool(n_jobs=num_cores,daemon=False) as pool:
        results = pool.map(run_musicChoreography_execution, results, progress_bar=progress_bar)
    return results








# uvicorn learnMultipleGameRules:app --reload

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
    
def run_gameRulesLearner_execution(game_topic:str,model:str)->str:
    role1 = "system"
    role2 = "user"
    ROLE = "role"
    CONTENT = "content"
    messages = [
        {ROLE:role1,CONTENT:f"Give clear game rules on how to approach this game. Keep it as short and concise and to the point as possible."},
        {ROLE:role2,CONTENT:game_topic},
    ]
    
    topic_response = get_response(model,messages,None)
    topic = get_choice(topic_response)
    return {"request":game_topic,"response":topic}

filename = os.path.basename(__file__).split(".")[0]
app = FastAPI(
    title=f"{filename}",
    version="1.0.0"
)


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/learnMultipleGameRules/")
def read_item(model:str,progress_bar:bool,game_topics:List[str] = Query(...)):
    num_cores = max(min(multiprocessing.cpu_count() // 2, 2),1)
    results = []
    for game_topic in game_topics:
        my_dict = {
            "game_topic":game_topic,
            "model":model
        }
        results.append(my_dict)
    with WorkerPool(n_jobs=num_cores,daemon=False) as pool:
        results = pool.map(run_gameRulesLearner_execution, results, progress_bar=progress_bar)
    return results













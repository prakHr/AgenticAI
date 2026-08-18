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
    
def run_planner_execution(goal:str,model:str)->str:
    
    role1 = "system"
    role2 = "user"
    ROLE = "role"
    CONTENT = "content"
    # model="openai/gpt-oss-120b"
    messages = [
        {ROLE:role1,CONTENT:"Break the goal into atomic steps. Return a JSON object with a key named 'steps' whose value is an array of strings. Example: {\"steps\": [\"Step 1\", \"Step 2\", \"Step 3\"]}."},
        {ROLE:role2,CONTENT:goal},
    ]
    response_format = {"type":"json_object"}
    plan_response = get_response(model,messages,response_format)
    plan = json.loads(get_choice(plan_response)).get("steps",[])

    results = []
    for steps in plan:
        messages = [
            {ROLE:role1,CONTENT:"Execute this single step and report the outcome concisely."},
            {ROLE:role2,CONTENT:steps},
        ]
        execution_response = get_response(model,messages,None)
        results.append({"step":steps,"result":get_choice(execution_response)})

    messages = [
        {ROLE:role1,CONTENT:"Summarize the completed plan and results for the user."},
        {ROLE:role2,CONTENT:json.dumps(results)},
    ]

    final_response = get_response(model,messages,None)
    return {"request":goal,"response":get_choice(final_response)}


app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/planner/{plan}")
def read_item(plan: str,model:str):
    planner_response = run_planner_execution(plan,model)
    return planner_response

@app.get("/planners/")
def read_item(model:str,progress_bar:bool,plan: List[str] = Query(...)):
    if len(plan)>1:
        return {"request":plan,"response":"Price will go up. It is free plan!"}
    num_cores = min(multiprocessing.cpu_count()//2,1)
    results = [{"goal" : g, "model" : model} for g in plan]
    with WorkerPool(n_jobs=num_cores,daemon=False) as pool:
        results = pool.map(run_planner_execution, results, progress_bar=progress_bar)
    return results

    


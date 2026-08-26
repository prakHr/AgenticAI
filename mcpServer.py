import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import json
from fastapi import FastAPI
import mpire
import os
os.environ["OMP_NUM_THREADS"] = "1"c
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

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi_mcp import FastApiMCP

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
    
def run_planner_execution(goal:str,model:str)->str:
    
    role1 = "system"
    role2 = "user"
    ROLE = "role"
    CONTENT = "content"
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
    

    

filename = os.path.basename(__file__).split(".")[0]
app = FastAPI(
    title=f"{filename}",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")


@app.get("/planner/{plan}",operation_id="get_planners")
def read_item(plan: str,model:str):
    planner_response = run_planner_execution(plan,model)
    return planner_response

@app.get("/planners/")
def read_item(model:str,progress_bar:bool,plan: List[str] = Query(...)):
    num_cores = max(min(multiprocessing.cpu_count() // 2, 2),1)
    results = [{"goal" : g, "model" : model} for g in plan]
    with WorkerPool(n_jobs=num_cores,daemon=False) as pool:
        results = pool.map(run_planner_execution, results, progress_bar=progress_bar)
    return results

if __name__=="__main__":
    mcp = FastApiMCP(app,include_operations = ["get_planners"])
    mcp.mount_http()
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8000)
# uvicorn giveGuidances:app --reload

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

from fastapi_mcp import FastApiMCP

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
    

def run_giveGuidance_execution(system_prompt:str,human_prompt:str,model:str)->str:
    role1 = "system"
    role2 = "user"
    ROLE = "role"
    CONTENT = "content"
    messages = [
        {ROLE:role1,CONTENT:f"{system_prompt}. Please do not hallucinate."},
        {ROLE:role2,CONTENT:human_prompt},
    ]
    
    topic_response = get_response(model,messages,None)
    topic = get_choice(topic_response)
    return {"request":human_prompt,"response":topic}

def get_app(filename):

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
    return app



filename = os.path.basename(__file__).split(".")[0]

app = get_app(filename)
mcp = FastApiMCP(app)
mcp.mount_http()

@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

@app.get("/giveGuidances/")
def read_item(model:str,system_prompt:str,progress_bar:bool,human_prompts:List[str] = Query(...)):
    num_cores = max(min(multiprocessing.cpu_count() // 2, 2),1)
    results = []
    for human_prompt in human_prompts:
        my_dict = {
            "system_prompt":system_prompt,
            "human_prompt":human_prompt,
            "model":model
        }
        results.append(my_dict)
    with WorkerPool(n_jobs=num_cores,daemon=False) as pool:
        results = pool.map(run_giveGuidance_execution, results, progress_bar=progress_bar)
    return results

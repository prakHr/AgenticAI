# uvicorn giveGuidance:app --reload

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
import uvicorn
    


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

def run_app(app,host,port):
    uvicorn.run(app, host=host, port=port)

filename = os.path.basename(__file__).split(".")[0]

app = get_app(filename)

@app.get("/", include_in_schema=False,operation_id="give_guidance")
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

@app.get("/giveGuidance/")
def read_item(model:str,system_prompt:str,human_prompt:str):
    results = run_giveGuidance_execution(system_prompt,human_prompt,model)
    return results


if __name__ == "__main__":
    mcp = FastApiMCP(app,include_operations = ["give_guidance"])
    mcp.mount_http()
    run_app(app,"localhost",8000)
    
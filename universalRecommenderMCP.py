# uvicorn universalRecommender:app --reload

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
    
def run_universalSuggestion_execution(genre:str,model:str,style:str)->str:
    role1 = "system"
    role2 = "user"
    ROLE = "role"
    CONTENT = "content"
    my_dict= {"genre":genre,"style":style}
    r = json.dumps(my_dict)
    messages = [
        {ROLE:role1,CONTENT:f"Suggest some of the best style according to this genre type. Please do not hallucinate."},
        {ROLE:role2,CONTENT:r},
    ]
    
    topic_response = get_response(model,messages,None)
    topic = get_choice(topic_response)
    return {"request":my_dict,"response":topic}


from fastapi import FastAPI
from fastapi.responses import RedirectResponse

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

@app.get("/universalRecommender/",operation_id="get_universalRecommender")
def read_item(model:str,genre:str,style:str):
    results = run_universalSuggestion_execution(genre,model,style)
    return results





if __name__=="__main__":
    mcp = FastApiMCP(app,include_operations = ["get_universalRecommender"])
    mcp.mount_http()
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8000)












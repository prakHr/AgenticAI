# uvicorn suggestUniversalDays:app --reload

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
    
def run_dayTaskDoer_execution(day:str,model:str)->str:
    role1 = "system"
    role2 = "user"
    ROLE = "role"
    CONTENT = "content"
    messages = [
        {ROLE:role1,CONTENT:f"Suggest some priority tasks to do in your free time on the basis of a particular day."},
        {ROLE:role2,CONTENT:day},
    ]
    
    topic_response = get_response(model,messages,None)
    topic = get_choice(topic_response)
    return {"request":day,"response":topic}
    


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


def check(gym_day,available_days):
    if gym_day not in available_days:
        return False
    return True
   
@app.get("/suggestUniversalDays/",operation_id="suggestUniversalDays")
def read_item(model:str,day:str):
    available_days = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]  
    if not check(day,available_days):
        return {"request":day,"response":f"Please input one of these days only :- {available_days}"}
    results = run_dayTaskDoer_execution(day,model)
    return results

        

if __name__=="__main__":
    mcp = FastApiMCP(app,include_operations = ["suggestUniversalDays"])
    mcp.mount_http()
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8000)











# uvicorn askAboutPlanetElement:app --reload

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
    

def run_askElementDetails_execution(element_name:str,planet_name:str,model:str)->str:
    role1 = "system"
    role2 = "user"
    ROLE = "role"
    CONTENT = "content"
    my_dict = {
        "element_name":element_name,
        "planet_name":planet_name
    }
    r = json.dumps(my_dict)
    messages = [
        {ROLE:role1,CONTENT:f"Tell me the brief and concise point to point details of element present in the body of the planet. Only give ans if it is a valid type of element. Please do not hallucinate."},
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


@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

@app.get("/askAboutPlanetElements_type1/")
def read_item(model:str,progress_bar:bool,planet_name:str,element_names:List[str] = Query(...)):
    num_cores = max(min(multiprocessing.cpu_count() // 2, 2),1)
    results = []
    for element_name in element_names:
        my_dict = {
            "element_name":element_name,
            "planet_name":planet_name,
            "model":model
        }
        results.append(my_dict)
    with WorkerPool(n_jobs=num_cores,daemon=False) as pool:
        results = pool.map(run_askElementDetails_execution, results, progress_bar=progress_bar)
    return results


@app.get("/askAboutPlanetElements_type2/")
def read_item(model:str,progress_bar:bool,element_name:str,planet_names:List[str] = Query(...)):
    num_cores = max(min(multiprocessing.cpu_count() // 2, 2),1)
    results = []
    for planet_name in planet_names:
        my_dict = {
            "element_name":element_name,
            "planet_name":planet_name,
            "model":model
        }
        results.append(my_dict)
    with WorkerPool(n_jobs=num_cores,daemon=False) as pool:
        results = pool.map(run_askElementDetails_execution, results, progress_bar=progress_bar)
    return results

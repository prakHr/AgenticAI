# uvicorn suggestMultipleGymExercises:app --reload

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
    
def run_gymExercisesDoer_execution(gym_day:str,model:str)->str:
    role1 = "system"
    role2 = "user"
    ROLE = "role"
    CONTENT = "content"
    messages = [
        {ROLE:role1,CONTENT:f"Suggest gym exercises on the basis of a particular gym day."},
        {ROLE:role2,CONTENT:gym_day},
    ]
    
    topic_response = get_response(model,messages,None)
    topic = get_choice(topic_response)
    return {"request":gym_day,"response":topic}
    
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


def check(gym_day,available_days):
    if gym_day not in available_days:
        return False
    return True
     
@app.get("/suggestMultipleGymExercises/")
def read_item(model:str,progress_bar:bool,gym_days:List[str] = Query(...)):
    available_days = ["Monday","Tuesday","Wednesday","Thursday","Friday"]    
    num_cores = max(min(multiprocessing.cpu_count() // 2, 2),1)
    results = []
    for gym_day in gym_days:
        if not check(gym_day,available_days):
            return {"request":gym_day,"response":f"Please input one of these days only :- {available_days}"}
        my_dict = {
            "gym_day":gym_day,
            "model":model
        }
        results.append(my_dict)
    with WorkerPool(n_jobs=num_cores,daemon=False) as pool:
        results = pool.map(run_gymExercisesDoer_execution, results, progress_bar=progress_bar)
    return results
        










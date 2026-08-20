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

def run_classification_execution(user_prompt:str,model:str,labels:list)->str:
    role1 = "system"
    role2 = "user"
    ROLE = "role"
    CONTENT = "content"
    messages = [
        {ROLE:role1,CONTENT:f"Classify the request into one of {labels}. Reply with just the label."},
        {ROLE:role2,CONTENT:user_prompt},
    ]
    
    route_response = get_response(model,messages,None)
    label = get_choice(route_response).strip().lower()
    return {"request":user_prompt,"response":label}

def run_storyCreation_execution(topic:str,model:str)->str:
    role1 = "system"
    role2 = "user"
    ROLE = "role"
    CONTENT = "content"
    messages = [
        {ROLE:role1,CONTENT:f"Create a story on the basis of this topic."},
        {ROLE:role2,CONTENT:topic},
    ]
    
    story_response = get_response(model,messages,None)
    story = get_choice(story_response).strip().lower()
    return {"request":topic,"response":story}

def run_universalCreation_execution(topic:str,model:str)->str:
    role1 = "system"
    role2 = "user"
    ROLE = "role"
    CONTENT = "content"
    messages = [
        {ROLE:role1,CONTENT:f"Create anything you like on the basis of this topic."},
        {ROLE:role2,CONTENT:topic},
    ]
    
    story_response = get_response(model,messages,None)
    story = get_choice(story_response).strip().lower()
    return {"request":topic,"response":story}

def run_nativeLanguagestoryCreation_execution(topic:str,model:str,language:str)->str:
    role1 = "system"
    role2 = "user"
    ROLE = "role"
    CONTENT = "content"
    messages = [
        {ROLE:role1,CONTENT:f"Create a story on the basis of this topic in this native language {language}."},
        {ROLE:role2,CONTENT:topic},
    ]
    
    story_response = get_response(model,messages,None)
    story = get_choice(story_response).strip().lower()
    return {"request":topic,"response":story,"language":language}

def run_topicCreation_execution(user_prompt:str,model:str)->str:
    role1 = "system"
    role2 = "user"
    ROLE = "role"
    CONTENT = "content"
    messages = [
        {ROLE:role1,CONTENT:f"Suggest only 1 topic in 2-3 words on the basis of this prompt."},
        {ROLE:role2,CONTENT:user_prompt},
    ]
    
    topic_response = get_response(model,messages,None)
    topic = get_choice(topic_response).strip().lower()
    return {"request":user_prompt,"response":topic}




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
    if len(plan)>3:
        return {"request":plan,"response":"Price will go up. It is free plan!"}
    num_cores = max(min(multiprocessing.cpu_count() // 2, 2),1)
    results = [{"goal" : g, "model" : model} for g in plan]
    with WorkerPool(n_jobs=num_cores,daemon=False) as pool:
        results = pool.map(run_planner_execution, results, progress_bar=progress_bar)
    return results

    
@app.get("/classify/")
def read_item(model:str,user_prompt:str,labels: List[str] = Query(...)):
    labels.append("other")
    labels = list(set([l.lower() for l in labels]))
    results = run_classification_execution(user_prompt,model,labels)
    return results

@app.get("/classifies/")
def read_item(model:str,progress_bar:bool,user_prompts: List[str] = Query(...),labels: List[str] = Query(...)):
    labels.append("other")
    labels = list(set([l.lower() for l in labels]))
    num_cores = max(min(multiprocessing.cpu_count() // 2, 2),1)
    results = []
    for user_prompt in user_prompts:
        my_dict = {
            "user_prompt":user_prompt,
            "model":model,
            "labels":labels
        }
        results.append(my_dict)
    with WorkerPool(n_jobs=num_cores,daemon=False) as pool:
        results = pool.map(run_classification_execution, results, progress_bar=progress_bar)
    return results

@app.get("/storyCreator/")
def read_item(model:str,topic:str):
    results = run_storyCreation_execution(topic,model)
    return results

@app.get("/nativeLanguageStoryCreator/")
def read_item(model:str,topic:str,language:str):
    results = run_nativeLanguagestoryCreation_execution(topic,model,language)
    return results

@app.get("/suggestTopic/")
def read_item(model:str,user_prompt:str):
    results = run_topicCreation_execution(user_prompt,model)
    return results

@app.get("/suggestTopics/")
def read_item(model:str,progress_bar:bool,user_prompts: List[str] = Query(...)):
    num_cores = max(min(multiprocessing.cpu_count() // 2, 2),1)
    results = []
    for user_prompt in user_prompts:
        my_dict = {
            "user_prompt":user_prompt,
            "model":model
        }
        results.append(my_dict)
    with WorkerPool(n_jobs=num_cores,daemon=False) as pool:
        results = pool.map(run_topicCreation_execution, results, progress_bar=progress_bar)
    
    return results

@app.get("/universalCreator/")
def read_item(model:str,user_prompt:str):
    results = run_universalCreation_execution(user_prompt,model)
    return results

# uvicorn planner:app --reload

import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import json
from fastapi import FastAPI

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key kaha hai bhai")
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
    return get_choice(final_response)


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/planner/{plan}")
def read_item(plan: str):
    model="openai/gpt-oss-120b"
    planner_response = run_planner_execution(plan,model)
    return {"request":plan,"response":planner_response}

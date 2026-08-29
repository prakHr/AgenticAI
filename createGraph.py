import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi_mcp import FastApiMCP
from groq import Groq

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key not found!")
client = Groq(api_key=my_api_key)

class State(TypedDict):
    goal: str
    response: str




def planner(state: State):
    default_model = "openai/gpt-oss-120b" 
    llm = ChatGroq(
        model=default_model,
        temperature=0.1,
    )

    response = llm.invoke(
        f"""
        Create a step-by-step plan for the following goal:

        {state["goal"]}
        """
    )

    return {
        "response": response.content
    }



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


def create_graph():
    
    graph = StateGraph(State)
    graph.add_node("planner", planner)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", END)
    return graph

def get_graph(graph):
    compiled_graph = graph.compile()
    return compiled_graph

@app.get("/planner/{goal}")
def run(goal: str, model: str = "openai/gpt-oss-120b",operation_id="get_planners"):
    graph = create_graph()
    compiled_graph = get_graph(graph)

    result = compiled_graph.invoke({
        "goal": goal
    })

    return {
        "goal": goal,
        "response": result["response"]
    }



if __name__=="__main__":
    mcp = FastApiMCP(app,include_operations = ["get_planners"])
    mcp.mount_http()
    import uvicorn
    uvicorn.run(app,host="localhost",port=8000)
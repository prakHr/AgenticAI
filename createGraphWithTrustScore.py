import os
from typing import TypedDict, Dict, Any, List

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi_mcp import FastApiMCP

from groq import Groq


# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found!")


# ============================================================
# GROQ CLIENT
# ============================================================

client = Groq(api_key=GROQ_API_KEY)


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

DEFAULT_MODEL = "openai/gpt-oss-120b"

# Minimum score required for a node to execute
MIN_TRUST = 60.0


# ============================================================
# NODE TRUST CONFIGURATION
# ============================================================
#
# Every node can have its own trust profile.
#
# The values are examples for your own trust framework.
# They are NOT official Nerq scores.
#
# Each factor is between 0 and 100.
#
# security
# reliability
# transparency
# dependency_health
# operational_risk
# data_sensitivity
#
# The final score is calculated using weighted factors.
# ============================================================

NODE_TRUST_CONFIG = {

    "planner": {
        "security": 90,
        "reliability": 90,
        "transparency": 85,
        "dependency_health": 90,
        "operational_risk": 90,
        "data_sensitivity": 90,
        "weights": {
            "security": 0.25,
            "reliability": 0.20,
            "transparency": 0.10,
            "dependency_health": 0.15,
            "operational_risk": 0.15,
            "data_sensitivity": 0.15
        }
    },


    "researcher": {
        "security": 85,
        "reliability": 85,
        "transparency": 80,
        "dependency_health": 85,
        "operational_risk": 80,
        "data_sensitivity": 75,
        "weights": {
            "security": 0.25,
            "reliability": 0.20,
            "transparency": 0.10,
            "dependency_health": 0.15,
            "operational_risk": 0.15,
            "data_sensitivity": 0.15
        }
    },


    "reviewer": {
        "security": 95,
        "reliability": 90,
        "transparency": 90,
        "dependency_health": 90,
        "operational_risk": 95,
        "data_sensitivity": 90,
        "weights": {
            "security": 0.25,
            "reliability": 0.20,
            "transparency": 0.10,
            "dependency_health": 0.15,
            "operational_risk": 0.15,
            "data_sensitivity": 0.15
        }
    }
}


# ============================================================
# LANGGRAPH STATE
# ============================================================

class State(TypedDict, total=False):

    # --------------------------------------------------------
    # User input
    # --------------------------------------------------------

    goal: str
    model: str

    # --------------------------------------------------------
    # Current node
    # --------------------------------------------------------

    current_node: str

    # --------------------------------------------------------
    # Trust configuration
    # --------------------------------------------------------

    min_trust: float

    # --------------------------------------------------------
    # Current node trust information
    # --------------------------------------------------------

    trust_score: float
    trust_grade: str
    trust_approved: bool
    trust_status: str

    # --------------------------------------------------------
    # Current node trust factors
    # --------------------------------------------------------

    trust_factors: Dict[str, float]

    # --------------------------------------------------------
    # Trust history for all nodes
    # --------------------------------------------------------

    trust_history: List[Dict[str, Any]]

    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    response: str


# ============================================================
# TRUST GRADE
# ============================================================

def calculate_grade(score: float) -> str:

    if score >= 95:
        return "A+"
    elif score >= 90:
        return "A"
    elif score >= 85:
        return "B+"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"


# ============================================================
# TRUST SCORE CALCULATION
# ============================================================

def calculate_trust_score(node_name: str) -> Dict[str, Any]:

    config = NODE_TRUST_CONFIG.get(node_name)

    # --------------------------------------------------------
    # If node has no configuration
    # --------------------------------------------------------

    if not config:
        return {
            "score": 0.0,
            "grade": "F",
            "factors": {},
            "error": (
                f"No trust configuration found "
                f"for node '{node_name}'"
            )
        }

    # --------------------------------------------------------
    # Factors
    # --------------------------------------------------------

    factors = {
        "security": config["security"],
        "reliability": config["reliability"],
        "transparency": config["transparency"],
        "dependency_health": config["dependency_health"],
        "operational_risk": config["operational_risk"],
        "data_sensitivity": config["data_sensitivity"]
    }

    weights = config["weights"]

    # --------------------------------------------------------
    # Weighted trust score
    # --------------------------------------------------------

    score = 0.0
    for factor, value in factors.items():
        weight = weights.get(factor,0)
        score += value * weight
    score = round(score,2)

    # --------------------------------------------------------
    # Grade
    # --------------------------------------------------------

    grade = calculate_grade(score)

    return {
        "score": score,
        "grade": grade,
        "factors": factors
    }


# ============================================================
# NODE TRUST CHECK
# ============================================================

def check_node_trust(
    node_name: str,
    state: State
):

    min_trust = state.get("min_trust",MIN_TRUST)
    trust = calculate_trust_score(node_name)
    score = trust["score"]
    grade = trust["grade"]
    factors = trust["factors"]

    # --------------------------------------------------------
    # Approval
    # --------------------------------------------------------

    approved = (score >= min_trust)

    if approved:status = "APPROVED"
    else:status = "REJECTED"

    # --------------------------------------------------------
    # Trust record
    # --------------------------------------------------------

    trust_record = {
        "node": node_name,
        "score": score,
        "grade": grade,
        "approved": approved,
        "status": status,
        "minimum_required": min_trust,
        "factors": factors
    }

    # --------------------------------------------------------
    # Preserve previous history
    # --------------------------------------------------------

    history = list(state.get("trust_history",[]))

    history.append(trust_record)

    # --------------------------------------------------------
    # Debug
    # --------------------------------------------------------

    # print()
    # print("=" * 70)
    # print("                    TRUST CHECK")
    # print("=" * 70)

    # print(
    #     "Node              :",
    #     node_name
    # )

    # print(
    #     "Trust Score       :",
    #     score
    # )

    # print(
    #     "Trust Grade       :",
    #     grade
    # )

    # print(
    #     "Minimum Required  :",
    #     min_trust
    # )

    # print(
    #     "Approved           :",
    #     approved
    # )

    # print(
    #     "Status             :",
    #     status
    # )

    # print()
    # print("Trust Factors:")

    # for factor, value in factors.items():

    #     print(
    #         f"  {factor:<22}: {value}"
    #     )

    # print("=" * 70)
    # print()

    return {
        "current_node": node_name,
        "trust_score": score,
        "trust_grade": grade,
        "trust_approved": approved,
        "trust_status": status,
        "trust_factors": factors,
        "trust_history": history
    }


# ============================================================
# PLANNER TRUST NODE
# ============================================================

def planner_trust_check(state: State):

    return check_node_trust("planner",state)


# ============================================================
# PLANNER NODE
# ============================================================

def planner(state: State):

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if not state.get("trust_approved",False):

        return {
            "current_node": "planner",
            "response": (
                "Planner execution blocked because "
                "the planner failed the trust check."
            )
        }

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = state.get("model",DEFAULT_MODEL)

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    llm = ChatGroq(model=model,temperature=0.1,api_key=GROQ_API_KEY)

    # --------------------------------------------------------
    # Generate plan
    # --------------------------------------------------------

    response = llm.invoke(

        f"""
Create a detailed step-by-step plan for the
following goal:

{state["goal"]}

Requirements:

1. Break the goal into logical steps.
2. Explain each step clearly.
3. Mention prerequisites.
4. Include practical implementation guidance.
5. Keep the plan structured and easy to follow.
"""
    )

    return {
        "current_node": "planner",
        "response": response.content
    }


# ============================================================
# PLANNER ROUTER
# ============================================================

def planner_router(state: State):

    if state.get("trust_approved",False):
        return "approved"
    return "rejected"


# ============================================================
# PLANNER REJECTION NODE
# ============================================================

def planner_rejected(state: State):
    score = state.get("trust_score")
    grade = state.get("trust_grade")

    minimum = state.get("min_trust",MIN_TRUST)

    return {
        "current_node":"planner_rejected",
        "response": (
            f"Planner execution blocked.\n\n"
            f"Trust Score: {score}\n"
            f"Trust Grade: {grade}\n"
            f"Minimum Required: {minimum}\n"
        )
    }


# ============================================================
# CREATE GRAPH
# ============================================================

def create_graph():
    graph = StateGraph(State)

    # --------------------------------------------------------
    # Trust node
    # --------------------------------------------------------

    graph.add_node("planner_trust_check",planner_trust_check)

    # --------------------------------------------------------
    # Planner
    # --------------------------------------------------------

    graph.add_node("planner",planner)

    # --------------------------------------------------------
    # Rejected
    # --------------------------------------------------------

    graph.add_node("planner_rejected",planner_rejected)

    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    graph.add_edge(START,"planner_trust_check")

    # --------------------------------------------------------
    # Trust routing
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "planner_trust_check",
        planner_router,
        {
            "approved":"planner",
            "rejected":"planner_rejected"
        }
    )

    # --------------------------------------------------------
    # END
    # --------------------------------------------------------

    graph.add_edge("planner",END)
    graph.add_edge("planner_rejected",END)
    return graph


# ============================================================
# COMPILE GRAPH
# ============================================================

def get_graph():
    graph = create_graph()
    return graph.compile()


# ============================================================
# FASTAPI APPLICATION
# ============================================================

filename = os.path.basename(__file__).split(".")[0]


app = FastAPI(title=filename,version="1.0.0")


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# ROOT -> SWAGGER
# ============================================================

@app.get("/",include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")


# ============================================================
# PLANNER API
# ============================================================

@app.get("/planner/{goal}",operation_id="get_planners")
def run(goal: str,model: str = DEFAULT_MODEL,min_trust: float = MIN_TRUST):

    # --------------------------------------------------------
    # Compile graph
    # --------------------------------------------------------

    compiled_graph = get_graph()

    # --------------------------------------------------------
    # Initial state
    # --------------------------------------------------------

    initial_state: State = {
        "goal": goal,
        "model": model,
        "current_node": "planner",
        "min_trust": min_trust,
        "trust_history": []
    }

    # --------------------------------------------------------
    # Execute
    # --------------------------------------------------------

    result = compiled_graph.invoke(initial_state)

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {
        "goal": goal,
        "model": model,
        "minimum_trust": min_trust,
        "current_node":result.get("current_node"),
        "trust": {
            "score":result.get("trust_score"),
            "grade":result.get("trust_grade"),
            "approved":result.get("trust_approved",False),
            "status":result.get("trust_status"),
            "factors":result.get("trust_factors",{})
        },
        "trust_history":result.get("trust_history",[]),
        "response":result.get("response")
    }


# ============================================================
# TRUST SCORE API
# ============================================================

@app.get("/trust/{node_name}")
def get_trust_score(node_name: str,min_trust: float = MIN_TRUST):
    trust = calculate_trust_score(node_name)
    score = trust["score"]
    approved = (score >= min_trust)
    return {
        "node":node_name,
        "trust": {
            "score":score,
            "grade":trust["grade"],
            "minimum_required":min_trust,
            "approved":approved,
            "status":
                "APPROVED"
                if approved
                else "REJECTED",
            "factors":trust["factors"]
        }
    }


# ============================================================
# ALL NODE TRUST SCORES API
# ============================================================

@app.get("/trust")
def get_all_trust_scores(min_trust: float = MIN_TRUST):
    results = []
    for node_name in NODE_TRUST_CONFIG:
        trust = calculate_trust_score(node_name)
        score = trust["score"]
        approved = (score >= min_trust)
        results.append({
            "node":node_name,
            "score":score,
            "grade":trust["grade"],
            "approved":approved,
            "status":
                "APPROVED"
                if approved
                else "REJECTED",
            "factors":trust["factors"]
        })

    return {
        "minimum_required":min_trust,
        "nodes":results
    }


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # FastAPI MCP
    # --------------------------------------------------------

    mcp = FastApiMCP(app,include_operations=["get_planners","get_trust_score","get_all_trust_scores"])
    mcp.mount_http()

    # --------------------------------------------------------
    # Uvicorn
    # --------------------------------------------------------

    import uvicorn

    uvicorn.run(app,host="0.0.0.0",port=8000)
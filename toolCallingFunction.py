import os
import json
import requests

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found!")

client = Groq(api_key=api_key)

model = "openai/gpt-oss-120b"


# ============================================================
# TOOL FUNCTION
# ============================================================

def search_web(query: str):
    """
    Search the web for a query and return relevant URLs.
    """

    search_url = "https://html.duckduckgo.com/html/"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/151.0 Safari/537.36"
        )
    }

    response = requests.get(
        search_url,
        params={"q": query},
        headers=headers,
        timeout=20,
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    results = []

    for result in soup.select(".result"):

        link = result.select_one(
            ".result__a"
        )

        if not link:
            continue

        title = link.get_text(
            " ",
            strip=True
        )

        url = link.get("href")

        if not url:
            continue

        results.append(
            {
                "title": title,
                "url": url,
            }
        )

    # Remove duplicate URLs
    unique_results = []
    seen = set()

    for result in results:

        if result["url"] not in seen:

            seen.add(result["url"])
            unique_results.append(result)

    return unique_results[:10]


# ============================================================
# TOOL DEFINITION
# ============================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",

            "description": """
            Search the web for a given query and return
            a list of relevant website URLs and titles.
            """,

            "parameters": {
                "type": "object",

                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The search query to find "
                            "relevant websites."
                        ),
                    }
                },

                "required": ["query"],
            },
        },
    }
]


# ============================================================
# USER QUERY
# ============================================================

query = input(
    "What do you want to search for? "
)


# ============================================================
# MESSAGES
# ============================================================

messages = [

    {
        "role": "system",
        "content": """
        You are a helpful web search assistant.

        When the user asks to find websites or URLs,
        use the search_web tool.

        Return the URLs found by the tool.
        Do not invent URLs.
        """,
    },

    {
        "role": "user",
        "content": query,
    },
]


# ============================================================
# FIRST MODEL REQUEST
# ============================================================

response = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=tools,
    tool_choice="auto",
    max_tokens=4096,
)


# ============================================================
# ASSISTANT RESPONSE
# ============================================================

response_message = response.choices[0].message

tool_calls = response_message.tool_calls or []


# ============================================================
# TOOL CALL
# ============================================================

if tool_calls:

    # Keep original assistant message
    messages.append(response_message)

    available_functions = {
        "search_web": search_web
    }

    for tool_call in tool_calls:

        function_name = (
            tool_call.function.name
        )

        function_to_call = (
            available_functions[
                function_name
            ]
        )

        # Parse arguments
        function_args = json.loads(
            tool_call.function.arguments
        )

        print(
            "\nTool arguments:"
        )

        print(
            function_args
        )

        # Execute function
        function_response = (
            function_to_call(
                **function_args
            )
        )

        print(
            "\nSearch results:"
        )

        print(
            json.dumps(
                function_response,
                indent=2
            )
        )

     
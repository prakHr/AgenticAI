import os
import json
import requests

from urllib.parse import urljoin
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
# TOOL 1: SEARCH WEB
# ============================================================

def search_web(query: str):
    """
    Search the web for a topic and return URLs.

    Replace the implementation of this function with
    your preferred search API.
    """

    # --------------------------------------------------------
    # Example using DuckDuckGo HTML search
    # --------------------------------------------------------

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

        href = link.get("href")

        if not href:
            continue

        # DuckDuckGo sometimes returns redirect URLs.
        # Extract the actual URL if necessary.
        if href.startswith("//"):
            href = "https:" + href

        results.append(
            {
                "title": title,
                "url": href,
            }
        )

    # Remove duplicates
    unique_results = []
    seen = set()

    for result in results:

        if result["url"] not in seen:

            seen.add(result["url"])
            unique_results.append(result)

    return unique_results[:5]


# ============================================================
# TOOL 2: SCRAPE WEBSITE
# ============================================================

def scrape_website(url: str):
    """
    Scrape a webpage and return its title and text.
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/151.0 Safari/537.36"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=20,
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    title = ""

    if soup.title:
        title = soup.title.get_text(
            strip=True
        )

    # --------------------------------------------------------
    # Remove unnecessary HTML
    # --------------------------------------------------------

    for tag in soup(
        [
            "script",
            "style",
            "noscript",
            "header",
            "footer",
            "nav",
        ]
    ):
        tag.decompose()

    # --------------------------------------------------------
    # Extract text
    # --------------------------------------------------------

    text = soup.get_text(
        separator=" ",
        strip=True
    )

    # Limit text sent to LLM
    text = text[:15000]

    # --------------------------------------------------------
    # Extract links
    # --------------------------------------------------------

    links = []

    for a in soup.find_all(
        "a",
        href=True
    ):

        href = a["href"]

        if href.startswith(
            (
                "#",
                "javascript:",
                "mailto:",
                "tel:",
            )
        ):
            continue

        absolute_url = urljoin(
            url,
            href
        )

        links.append(
            absolute_url
        )

    links = list(
        dict.fromkeys(links)
    )

    return {
        "url": url,
        "title": title,
        "text": text,
        "links": links[:100],
    }


# ============================================================
# TOOL DEFINITIONS
# ============================================================

tools = [

    # --------------------------------------------------------
    # SEARCH TOOL
    # --------------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "search_web",

            "description": """
            Search the web for a topic and return
            relevant URLs and titles.
            """,

            "parameters": {
                "type": "object",

                "properties": {

                    "query": {
                        "type": "string",
                        "description": (
                            "The topic or search query."
                        ),
                    }

                },

                "required": ["query"],
            },
        },
    },


    # --------------------------------------------------------
    # SCRAPER TOOL
    # --------------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "scrape_website",

            "description": """
            Scrape a webpage and return its title,
            textual content and links.
            """,

            "parameters": {
                "type": "object",

                "properties": {

                    "url": {
                        "type": "string",
                        "description": (
                            "The URL of the webpage to scrape."
                        ),
                    }

                },

                "required": ["url"],
            },
        },
    },
]


# ============================================================
# AVAILABLE FUNCTIONS
# ============================================================

available_functions = {
    "search_web": search_web,
    "scrape_website": scrape_website,
}


# ============================================================
# USER TOPIC
# ============================================================

topic = input(
    "Enter a topic to research: "
)


# ============================================================
# INITIAL MESSAGE
# ============================================================

messages = [

    {
        "role": "system",

        "content": """
You are a web research assistant.

Your job is to research a topic using tools.

Follow this workflow:

1. Use search_web to find relevant URLs.
2. Examine the search results.
3. Select the most relevant URLs.
4. Use scrape_website to scrape those URLs.
5. Analyze the scraped content.
6. Give the user a clear research summary.

Do not invent information.

Use the scraped website content as the
primary source for your answer.

Prefer authoritative and relevant websites.
""",
    },

    {
        "role": "user",

        "content": f"""
Research this topic:

{topic}

Find relevant websites, scrape the most
useful pages, and summarize the information.
""",
    },
]


# ============================================================
# AGENT LOOP
# ============================================================

while True:

    response = client.chat.completions.create(

        model=model,

        messages=messages,

        tools=tools,

        tool_choice="auto",

        max_tokens=4096,
    )

    response_message = response.choices[0].message

    tool_calls = (
        response_message.tool_calls or []
    )


    # --------------------------------------------------------
    # No more tools -> final answer
    # --------------------------------------------------------

    if not tool_calls:

        print(
            "\n\n=============================="
        )

        print("FINAL ANSWER")

        print(
            "==============================\n"
        )

        print(
            response_message.content
        )

        break


    # --------------------------------------------------------
    # Preserve assistant message
    # --------------------------------------------------------

    messages.append(
        response_message
    )


    # --------------------------------------------------------
    # Execute tools
    # --------------------------------------------------------

    for tool_call in tool_calls:

        function_name = (
            tool_call.function.name
        )

        function_args = json.loads(
            tool_call.function.arguments
        )

        print(
            f"\nCalling tool: "
            f"{function_name}"
        )

        print(
            f"Arguments: "
            f"{function_args}"
        )


        # ----------------------------------------------------
        # Find Python function
        # ----------------------------------------------------

        function_to_call = (
            available_functions[
                function_name
            ]
        )


        # ----------------------------------------------------
        # Execute
        # ----------------------------------------------------

        try:

            function_response = (
                function_to_call(
                    **function_args
                )
            )

        except Exception as e:

            function_response = {
                "error": str(e)
            }


        # ----------------------------------------------------
        # Send tool result to LLM
        # ----------------------------------------------------

        messages.append(
            {
                "role": "tool",

                "content": json.dumps(
                    function_response
                ),

                "tool_call_id":
                    tool_call.id,
            }
        )
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

from pprint import pprint

import mpire
os.environ["OMP_NUM_THREADS"] = "1"
import time
import multiprocessing 
from mpire import WorkerPool
from pprint import pprint
from multiprocessing import Manager
from transformers import pipeline, AutoTokenizer



from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

import json
import requests
import multiprocessing

from urllib.parse import urljoin, urlparse, parse_qs

from bs4 import BeautifulSoup




# ============================================================
# CONFIGURATION
# ============================================================

os.environ["OMP_NUM_THREADS"] = "1"

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found!")

client = Groq(api_key=api_key)


def get_choice(response):
    return response.choices[0].message.content

def get_response(model,messages,response_format):
    if response_format:
        return client.chat.completions.create(model=model, messages=messages, response_format=response_format)
    return client.chat.completions.create(model=model, messages=messages)

def call(system:str,user:str,model:str)->str:
    messages = [
        {
            "role":"system",
            "content":system
        },
        {
            "role":"user",
            "content":user
        }
    ]
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content

    
def research_team(task:str,model:str)->str:
    facts = call("You are a research worker. Return 3 concise bullet facts.",task,model)
    return call("You are the researcher team lead. Consolidate the bullets into one paragraph.",facts,model)


def writing_team(task:str,model:str)->str:
    draft = call("You are a writing worker. Draft a short paragraph.",task,model)
    return call("You are the writing team lead. Edit for clarity and tone.",draft,model)

def run_hierarchical(goal:str,model:str)->str:
    r = research_team(goal,model)
    draft = writing_team(f"Goal {goal}\nResearch:{r}",model)
    return call("You are the top-level manager. Approve or refine the final output.",draft,model)
# ============================================================
# COMMON HEADERS
# ============================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/151.0 Safari/537.36"
    )
}


# ============================================================
# TOOL FUNCTION
# ============================================================

def search_web(query: str):
    """
    Search DuckDuckGo and return relevant URLs and titles.
    """

    search_url = "https://html.duckduckgo.com/html/"

    response = requests.get(
        search_url,
        params={"q": query},
        headers=HEADERS,
        timeout=20,
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    results = []

    for result in soup.select(".result"):

        link = result.select_one(".result__a")

        if not link:
            continue

        title = link.get_text(
            " ",
            strip=True
        )

        url = link.get("href")

        if not url:
            continue

        # ====================================================
        # FIX DUCKDUCKGO REDIRECT URL
        # ====================================================

        if url.startswith("//duckduckgo.com/l/"):

            parsed_url = urlparse(
                "https:" + url
            )

            query_params = parse_qs(
                parsed_url.query
            )

            if "uddg" in query_params:

                url = query_params["uddg"][0]

        # ====================================================
        # ONLY ACCEPT HTTP/HTTPS URLS
        # ====================================================

        if not url.startswith(
            ("http://", "https://")
        ):
            continue

        results.append(
            {
                "title": title,
                "url": url,
            }
        )

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    unique_results = []

    seen = set()

    for result in results:

        if result["url"] in seen:
            continue

        seen.add(
            result["url"]
        )

        unique_results.append(
            result
        )

    return unique_results


# ============================================================
# TOOL 2: SCRAPE WEBSITE
# ============================================================

def scrape_website(url):
    """
    Scrape a webpage and return:
    - URL
    - title
    - text
    - links
    """


    # print(
    #     f"Scraping: {url}"
    # )

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=20,
            allow_redirects=True,
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # ====================================================
        # TITLE
        # ====================================================

        title = ""

        if soup.title:

            title = soup.title.get_text(
                strip=True
            )

        # ====================================================
        # REMOVE UNNECESSARY HTML
        # ====================================================

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

        # ====================================================
        # EXTRACT TEXT
        # ====================================================

        text = soup.get_text(
            separator=" ",
            strip=True
        )

        # Limit text
        text = text[:15000]

        # ====================================================
        # EXTRACT LINKS
        # ====================================================

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

            if absolute_url.startswith(
                ("http://", "https://")
            ):

                links.append(
                    absolute_url
                )

        # ====================================================
        # REMOVE DUPLICATE LINKS
        # ====================================================

        links = list(
            dict.fromkeys(links)
        )

        return {
            "url": url,
            "title": title,
            "text": text,
            "links": links[:100],
            "error": None,
        }

    except Exception as e:

        # print(
        #     f"ERROR scraping {url}: {e}"
        # )

        return {
            "url": url,
            "title": "",
            "text": "",
            "links": [],
            "error": str(e),
        }


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

                "required": [
                    "query"
                ],
            },
        },
    }
]


def load_big_model(worker_state):

    model_name = "facebook/bart-large-cnn"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    worker_state["tokenizer"] = tokenizer

    worker_state["summarizer"] = pipeline(
        "summarization",
        model=model_name,
        tokenizer=tokenizer,
        device=-1
    )    


def chunk_text(tokenizer, text, chunk_size=700):

    tokens = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    chunks = []

    for i in range(0, len(tokens), chunk_size):

        chunk_tokens = tokens[i:i + chunk_size]

        chunk = tokenizer.decode(
            chunk_tokens,
            skip_special_tokens=True
        )

        if chunk.strip():
            chunks.append(chunk)

    return chunks


def summarize(worker_state, page_content):

    summarizer = worker_state["summarizer"]
    tokenizer = worker_state["tokenizer"]

    text = page_content.strip()

    if not text:
        return ""

    chunks = chunk_text(
        tokenizer,
        text,
        chunk_size=700
    )

    summaries = []

    for chunk in chunks:

        input_length = len(chunk.split())

        max_len = min(
            100,
            max(30, int(input_length * 0.6))
        )

        min_len = min(
            30,
            max(10, int(max_len * 0.4))
        )

        if min_len >= max_len:
            min_len = max(5, max_len // 2)

        summary = summarizer(
            chunk,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
            truncation=True
        )

        summaries.append(
            summary[0]["summary_text"]
        )


    return " ".join(summaries)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # USER QUERY
    # ========================================================
    model = "openai/gpt-oss-120b"

    query = input(
        "What do you want to search for? "
    )

    # ========================================================
    # MESSAGES
    # ========================================================

    messages = [

        {
            "role": "system",

            "content": """
            You are a helpful web search assistant.

            When the user asks to find websites or URLs,
            use the search_web tool.

            Convert the user's topic into a useful search query.

            Return the URLs found by the tool.

            Do not invent URLs.
            """
        },

        {
            "role": "user",

            "content": query,
        },
    ]

    # ========================================================
    # FIRST MODEL REQUEST
    # ========================================================

    print(
        "\nCalling LLM..."
    )

    response = client.chat.completions.create(

        model=model,

        messages=messages,

        tools=tools,

        tool_choice="auto",

        max_tokens=4096,
    )

    # ========================================================
    # ASSISTANT RESPONSE
    # ========================================================

    response_message = (
        response.choices[0].message
    )

    tool_calls = (
        response_message.tool_calls
        or []
    )

    # ========================================================
    # RESULTS
    # ========================================================

    results = []

    # ========================================================
    # TOOL CALL
    # ========================================================

    if tool_calls:

        messages.append(
            response_message
        )

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

            # ================================================
            # PARSE ARGUMENTS
            # ================================================

            function_args = json.loads(
                tool_call.function.arguments
            )

            print(
                "\nTool arguments:"
            )

            print(
                function_args
            )

            # ================================================
            # EXECUTE TOOL
            # ================================================

            function_response = (
                function_to_call(
                    **function_args
                )
            )

            # ================================================
            # PRINT SEARCH RESULTS
            # ================================================

            print(
                "\nSearch results:"
            )

            for result in function_response:

                print(
                    result
                )

                results.append(
                    {
                        "url": result["url"]
                    }
                )

    else:


        print(
            "\nThe model did not call search_web."
        )
        
    # ========================================================
    # SHOW URLS
    # ========================================================

    print(
        "\nURLs found:"
    )

    for item in results:

        print(
            item["url"]
        )


    # ========================================================
    # STOP IF NO URLS
    # ========================================================

    if not results:

        print(
            "\nNo URLs found."
        )

        exit()


    # ========================================================
    # NUMBER OF WORKERS
    # ========================================================

    num_cores = max(
        min(
            multiprocessing.cpu_count() // 2,
            2
        ),
        1
    )

    print(
        f"\nUsing {num_cores} workers"
    )


    # ========================================================
    # SCRAPE WEBSITES
    # ========================================================

    print(
        "\nStarting scraping..."
    )

    with WorkerPool(
        n_jobs=num_cores,
        daemon=False
    ) as pool:

        scraped_results = pool.map(
            scrape_website,
            results,
            progress_bar=True
        )


    # ========================================================
    # SCRAPING COMPLETE
    # ========================================================

    print(
        "\nhere3"
    )

    print(
        "\nScraping completed!"
    )


    # ========================================================
    # PRINT SCRAPED RESULTS
    # ========================================================
    results2 = []
    for result in scraped_results:

        print(
            "\n========================================"
        )

        print(
            "URL:",
            result["url"]
        )

        print(
            "TITLE:",
            result["title"]
        )

        print(
            "ERROR:",
            result["error"]
        )

        print(
            "TEXT:"
        )

        print(
            result["text"][:1000]
        )

        print(
            "LINKS:",
            len(result["links"])
        )
        my_dict = {
            "page_content":result["text"]
            
        }
        results2.append(my_dict)

    
    
    # final_ans =  run_hierarchical(text,model)

    
    results = results2
        
        
    with WorkerPool(n_jobs=num_cores,daemon=False,use_worker_state=True) as pool:
        results = pool.map(summarize, results, progress_bar = True, worker_init=load_big_model)
    print("results = ")
    print(results)
    if len(results)>0:
        text = "".join(results)
        final_ans =  run_hierarchical(text,model)
        print("final_ans = ")
        print(final_ans)

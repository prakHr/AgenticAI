import os
import json
import requests
import multiprocessing

from urllib.parse import urljoin, urlparse, parse_qs

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq
from mpire import WorkerPool


# ============================================================
# CONFIGURATION
# ============================================================

os.environ["OMP_NUM_THREADS"] = "1"

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found!")

client = Groq(api_key=api_key)

model = "openai/gpt-oss-120b"


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


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # USER QUERY
    # ========================================================

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


    
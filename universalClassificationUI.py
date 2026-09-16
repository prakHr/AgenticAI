import os
import time
import json
import multiprocessing

from dotenv import load_dotenv
from groq import Groq

from dash import (
    Dash,
    html,
    dcc,
    Input,
    Output,
    State,
    callback,
    dash_table,
    no_update,
)

import dash_bootstrap_components as dbc


# ============================================================
# Configuration
# ============================================================

load_dotenv()

MY_API_KEY = os.getenv("GROQ_API_KEY")

if not MY_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Add it to your .env file."
    )

client = Groq(api_key=MY_API_KEY)


DEFAULT_MODEL = "openai/gpt-oss-120b"


# ============================================================
# Classification functions
# ============================================================

def get_choice(response):
    return response.choices[0].message.content


def get_response(model, messages, response_format=None):

    if response_format:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            response_format=response_format,
        )

    return client.chat.completions.create(
        model=model,
        messages=messages,
    )


def run_classification_execution(
    user_prompt: str,
    model: str,
    labels: list,
):

    labels = list(
        set(
            [
                str(label).lower().strip()
                for label in labels
                if str(label).strip()
            ]
        )
    )

    if "other" not in labels:
        labels.append("other")

    messages = [
        {
            "role": "system",
            "content": (
                f"Classify the request into one of {labels}. "
                "Reply with just the label."
            ),
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    response = get_response(
        model=model,
        messages=messages,
        response_format=None,
    )

    label = get_choice(response).strip().lower()

    return {
        "request": user_prompt,
        "response": label,
    }


def classify_multiple(
    user_prompts,
    model,
    labels,
):

    results = []

    for prompt in user_prompts:

        start_time = time.perf_counter()

        result = run_classification_execution(
            user_prompt=prompt,
            model=model,
            labels=labels,
        )

        elapsed = time.perf_counter() - start_time

        result["time"] = round(elapsed, 3)

        results.append(result)

    return results


# ============================================================
# Dash application
# ============================================================

app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG
    ],
    title="Universal Classification",
)

server = app.server


# ============================================================
# Styles
# ============================================================

CARD_STYLE = {
    "backgroundColor": "#1e1e1e",
    "border": "1px solid #333",
    "borderRadius": "14px",
    "padding": "24px",
    "marginBottom": "20px",
}

INPUT_STYLE = {
    "backgroundColor": "#111",
    "color": "white",
    "border": "1px solid #444",
    "borderRadius": "8px",
}


# ============================================================
# Layout
# ============================================================

app.layout = dbc.Container(
    [

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        dbc.Row(
            dbc.Col(
                [
                    html.H1(
                        "Universal Classification",
                        className="fw-bold",
                    ),

                    html.P(
                        "Classify user requests using Groq LLMs",
                        className="text-secondary",
                    ),
                ],
                width=12,
            ),
            className="mt-4 mb-4",
        ),


        # ----------------------------------------------------
        # Configuration
        # ----------------------------------------------------

        dbc.Row(
            dbc.Col(
                dbc.Card(
                    [
                        html.H4(
                            "Classification Configuration",
                            className="mb-4",
                        ),

                        # Model
                        html.Label(
                            "Model",
                            className="fw-bold mb-2",
                        ),

                        dcc.Dropdown(
                            id="model-dropdown",
                            options=[
                                {
                                    "label": "GPT-OSS 120B",
                                    "value": "openai/gpt-oss-120b",
                                },
                                {
                                    "label": "GPT-OSS 20B",
                                    "value": "openai/gpt-oss-20b",
                                },
                            ],
                            value=DEFAULT_MODEL,
                            clearable=False,
                            style={
                                "color": "#111",
                                "marginBottom": "20px",
                            },
                        ),

                        # Labels
                        html.Label(
                            "Classification Labels",
                            className="fw-bold mb-2",
                        ),

                        dbc.InputGroup(
                            [
                                dbc.Input(
                                    id="label-input",
                                    placeholder="Enter label e.g. billing",
                                    type="text",
                                    style=INPUT_STYLE,
                                ),

                                dbc.Button(
                                    "Add Label",
                                    id="add-label-btn",
                                    color="primary",
                                ),
                            ],
                            className="mb-3",
                        ),

                        html.Div(
                            id="labels-container",
                            className="d-flex flex-wrap gap-2",
                        ),

                    ],
                    style=CARD_STYLE,
                    color="dark",
                ),
                width=12,
            ),
        ),


        # ----------------------------------------------------
        # Single Classification
        # ----------------------------------------------------

        dbc.Row(
            dbc.Col(
                dbc.Card(
                    [

                        html.Div(
                            [
                                html.H4(
                                    "Single Classification"
                                ),

                                html.Span(
                                    "1",
                                    className=(
                                        "badge bg-primary "
                                        "float-end"
                                    ),
                                ),
                            ]
                        ),

                        html.P(
                            "Enter a request and classify it.",
                            className="text-secondary",
                        ),

                        dcc.Textarea(
                            id="single-prompt",
                            placeholder=(
                                "Example:\n"
                                "I am unable to login to my account."
                            ),
                            style={
                                **INPUT_STYLE,
                                "width": "100%",
                                "height": "140px",
                                "padding": "12px",
                            },
                        ),

                        dbc.Button(
                            "Classify",
                            id="classify-btn",
                            color="primary",
                            className="mt-3",
                        ),

                        dcc.Loading(
                            id="single-loading",
                            type="circle",
                            children=html.Div(
                                id="single-result",
                                className="mt-4",
                            ),
                        ),

                    ],
                    style=CARD_STYLE,
                    color="dark",
                ),
                width=12,
            ),
        ),


        # ----------------------------------------------------
        # Bulk Classification
        # ----------------------------------------------------

        dbc.Row(
            dbc.Col(
                dbc.Card(
                    [

                        html.Div(
                            [
                                html.H4(
                                    "Bulk Classification"
                                ),

                                html.Span(
                                    "2",
                                    className=(
                                        "badge bg-success "
                                        "float-end"
                                    ),
                                ),
                            ]
                        ),

                        html.P(
                            "Enter one request per line.",
                            className="text-secondary",
                        ),

                        dcc.Textarea(
                            id="bulk-prompts",
                            placeholder=(
                                "I cannot login\n"
                                "I want to upgrade my plan\n"
                                "My payment failed\n"
                                "I want to contact support"
                            ),
                            style={
                                **INPUT_STYLE,
                                "width": "100%",
                                "height": "220px",
                                "padding": "12px",
                            },
                        ),

                        dbc.Row(
                            [

                                dbc.Col(
                                    dbc.Button(
                                        "Classify All",
                                        id="bulk-classify-btn",
                                        color="success",
                                        className="mt-3",
                                    ),
                                    width="auto",
                                ),

                                dbc.Col(
                                    html.Div(
                                        id="bulk-count",
                                        className=(
                                            "text-secondary "
                                            "mt-3"
                                        ),
                                    ),
                                    width="auto",
                                ),

                            ]
                        ),

                        dcc.Loading(
                            id="bulk-loading",
                            type="circle",
                            children=html.Div(
                                id="bulk-results",
                                className="mt-4",
                            ),
                        ),

                    ],
                    style=CARD_STYLE,
                    color="dark",
                ),
                width=12,
            ),
        ),


        # ----------------------------------------------------
        # Footer
        # ----------------------------------------------------

        dbc.Row(
            dbc.Col(
                html.P(
                    "Universal Classification • Powered by Groq",
                    className=(
                        "text-center text-secondary "
                        "mt-4 mb-4"
                    ),
                )
            )
        ),

        # Hidden store
        dcc.Store(
            id="labels-store",
            data=[
                "billing",
                "technical",
                "sales",
                "support",
            ],
        ),

    ],
    fluid=True,
    style={
        "maxWidth": "1400px",
    },
)


# ============================================================
# Label management
# ============================================================

@callback(
    Output("labels-store", "data"),
    Output("label-input", "value"),

    Input("add-label-btn", "n_clicks"),

    State("label-input", "value"),
    State("labels-store", "data"),

    prevent_initial_call=True,
)
def add_label(n_clicks, label, labels):

    if not label:
        return labels, ""

    label = label.strip().lower()

    if label and label not in labels:
        labels.append(label)

    return labels, ""


@callback(
    Output("labels-container", "children"),

    Input("labels-store", "data"),
)
def display_labels(labels):

    if not labels:
        return html.Span(
            "No labels added",
            className="text-secondary",
        )

    components = []

    for label in labels:

        components.append(
            dbc.Badge(
                label,
                color="primary",
                className="p-2",
            )
        )

    components.append(
        dbc.Badge(
            "other",
            color="secondary",
            className="p-2",
        )
    )

    return components


# ============================================================
# Single classification callback
# ============================================================

@callback(
    Output("single-result", "children"),

    Input("classify-btn", "n_clicks"),

    State("single-prompt", "value"),
    State("model-dropdown", "value"),
    State("labels-store", "data"),

    prevent_initial_call=True,
)
def classify_single(
    n_clicks,
    prompt,
    model,
    labels,
):

    if not prompt or not prompt.strip():

        return dbc.Alert(
            "Please enter a request.",
            color="warning",
        )

    start_time = time.perf_counter()

    try:

        result = run_classification_execution(
            user_prompt=prompt.strip(),
            model=model,
            labels=labels,
        )

        elapsed = time.perf_counter() - start_time

        return dbc.Card(
            dbc.CardBody(
                [

                    html.Div(
                        [
                            html.Span(
                                "Classification",
                                className="text-secondary",
                            ),

                            html.H2(
                                result["response"],
                                className="mt-2",
                            ),
                        ]
                    ),

                    html.Hr(),

                    html.Div(
                        [
                            html.Span(
                                "Response time: ",
                                className="text-secondary",
                            ),

                            html.Span(
                                f"{elapsed:.2f}s"
                            ),
                        ]
                    ),

                ]
            ),
            color="dark",
            style={
                "border": "1px solid #444",
                "borderRadius": "10px",
            },
        )

    except Exception as e:

        return dbc.Alert(
            f"Classification failed: {str(e)}",
            color="danger",
        )


# ============================================================
# Bulk classification callback
# ============================================================

@callback(
    Output("bulk-results", "children"),
    Output("bulk-count", "children"),

    Input("bulk-classify-btn", "n_clicks"),

    State("bulk-prompts", "value"),
    State("model-dropdown", "value"),
    State("labels-store", "data"),

    prevent_initial_call=True,
)
def classify_bulk(
    n_clicks,
    prompts,
    model,
    labels,
):

    if not prompts:

        return (
            dbc.Alert(
                "Please enter at least one request.",
                color="warning",
            ),
            "",
        )

    user_prompts = [
        line.strip()
        for line in prompts.splitlines()
        if line.strip()
    ]

    if not user_prompts:

        return (
            dbc.Alert(
                "No valid prompts found.",
                color="warning",
            ),
            "",
        )

    try:

        start_time = time.perf_counter()

        results = classify_multiple(
            user_prompts=user_prompts,
            model=model,
            labels=labels,
        )

        total_time = time.perf_counter() - start_time

        table_data = []

        for index, result in enumerate(results, start=1):

            table_data.append(
                {
                    "No.": index,
                    "Request": result["request"],
                    "Classification": result["response"],
                    "Time (s)": result["time"],
                }
            )

        table = dash_table.DataTable(

            data=table_data,

            columns=[
                {
                    "name": "No.",
                    "id": "No.",
                },
                {
                    "name": "Request",
                    "id": "Request",
                },
                {
                    "name": "Classification",
                    "id": "Classification",
                },
                {
                    "name": "Time (s)",
                    "id": "Time (s)",
                },
            ],

            style_table={
                "overflowX": "auto",
            },

            style_header={
                "backgroundColor": "#111",
                "color": "white",
                "fontWeight": "bold",
                "border": "1px solid #444",
            },

            style_cell={
                "backgroundColor": "#1e1e1e",
                "color": "white",
                "border": "1px solid #333",
                "padding": "12px",
                "textAlign": "left",
                "whiteSpace": "normal",
                "height": "auto",
            },

            style_data_conditional=[

                {
                    "if": {
                        "column_id": "Classification"
                    },
                    "fontWeight": "bold",
                }

            ],

            page_size=10,

        )

        summary = dbc.Row(
            [

                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Small(
                                    "Requests",
                                    className="text-secondary",
                                ),
                                html.H3(
                                    len(results)
                                ),
                            ]
                        ),
                        color="dark",
                    ),
                    md=3,
                ),

                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Small(
                                    "Total Time",
                                    className="text-secondary",
                                ),
                                html.H3(
                                    f"{total_time:.2f}s"
                                ),
                            ]
                        ),
                        color="dark",
                    ),
                    md=3,
                ),

                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Small(
                                    "Avg Time",
                                    className="text-secondary",
                                ),
                                html.H3(
                                    f"{total_time / len(results):.2f}s"
                                ),
                            ]
                        ),
                        color="dark",
                    ),
                    md=3,
                ),

            ],
            className="mb-4",
        )

        return (
            html.Div(
                [
                    summary,
                    table,
                ]
            ),
            f"{len(results)} requests classified",
        )

    except Exception as e:

        return (
            dbc.Alert(
                f"Bulk classification failed: {str(e)}",
                color="danger",
            ),
            "",
        )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=8050,
    )
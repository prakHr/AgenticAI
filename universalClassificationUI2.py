import os
import time

from dotenv import load_dotenv

from typesafe_sdk import (
    TypeSafeClient,
    Choice,
)

from dash import (
    Dash,
    html,
    dcc,
    Input,
    Output,
    State,
    callback,
    dash_table,
)

import dash_bootstrap_components as dbc

from rapidfuzz import process, fuzz, utils


# ============================================================
# Configuration
# ============================================================

load_dotenv()

TYPESAFE_API_KEY = os.getenv("TYPESAFE_API_KEY")

if not TYPESAFE_API_KEY:
    raise ValueError(
        "TYPESAFE_API_KEY not found. "
        "Add it to your .env file."
    )


client = TypeSafeClient(
    api_key=TYPESAFE_API_KEY
)


DEFAULT_MODEL = "jev-latest"


# ============================================================
# Jev Classification
# ============================================================

def normalize_labels(labels):
    """
    Clean and normalize user-defined classification labels.
    """

    normalized = []

    for label in labels:

        label = str(label).strip().lower()

        if label and label not in normalized:
            normalized.append(label)

    return normalized


def run_classification_execution(
    user_prompt: str,
    labels: list,
):
    """
    Classify one request using Jev Choice.
    """

    labels = normalize_labels(labels)

    if not labels:
        raise ValueError(
            "At least one classification label is required."
        )

    if "other" not in labels:
        labels.append("other")

    # --------------------------------------------------------
    # Input state
    # --------------------------------------------------------

    state = {
        "request": user_prompt
    }

    # --------------------------------------------------------
    # Jev Choice
    # --------------------------------------------------------

    questions = {
        "classification": Choice(
            criteria={
                label: label
                for label in labels
            }
        )
    }

    # --------------------------------------------------------
    # Jev inference
    # --------------------------------------------------------

    response = client.system_one(
        state=state,
        questions=questions,
    )

    # --------------------------------------------------------
    # ChoiceAnswer
    # --------------------------------------------------------

    classification = response.answers["classification"]

    # ChoiceAnswer fields:
    #
    # classification.choice
    # classification.confidence
    # classification.probabilities

    label = classification.choice

    confidence = classification.confidence

    probabilities = classification.probabilities

    return {
        "request": user_prompt,
        "response": label,
        "confidence": confidence,
        "probabilities": probabilities,
    }

# ============================================================
# Bulk Classification
# ============================================================

def classify_multiple(
    user_prompts,
    labels,
):
    """
    Classify multiple requests.

    Each request is currently sent as one Jev decision.
    """

    results = []

    for prompt in user_prompts:

        start_time = time.perf_counter()

        result = run_classification_execution(
            user_prompt=prompt,
            labels=labels,
        )

        elapsed = time.perf_counter() - start_time

        result["time"] = round(
            elapsed,
            3
        )

        results.append(result)

    return results


# ============================================================
# Dash Application
# ============================================================

app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG
    ],
    title="Universal Classification — Jev",
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
                        "Fast typed classification using Jev by TypeSafe AI",
                        className="text-secondary",
                    ),

                    dbc.Badge(
                        "Powered by Jev • TypeSafe AI",
                        color="primary",
                        className="p-2 mt-2",
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

                        dbc.Input(
                            id="model-name",
                            value=DEFAULT_MODEL,
                            disabled=True,
                            style={
                                **INPUT_STYLE,
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
                                    placeholder=(
                                        "Enter label e.g. billing"
                                    ),
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
                            className=(
                                "d-flex flex-wrap gap-2"
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
                            "Enter a request and classify it using Jev.",
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
                                            "text-secondary mt-3"
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
                    "Universal Classification • Powered by Jev / TypeSafe AI",
                    className=(
                        "text-center text-secondary "
                        "mt-4 mb-4"
                    ),
                )
            )
        ),


        # ----------------------------------------------------
        # Hidden Store
        # ----------------------------------------------------

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
# Fuzzy Duplicate Detection
# ============================================================

def check(labels, label):

    if not labels:
        return True

    first_same_words = process.extractOne(
        label,
        labels,
        scorer=fuzz.WRatio,
        processor=utils.default_process,
    )

    if not first_same_words:
        return True

    first_same_word, score = (
        first_same_words[0],
        first_same_words[1],
    )

    THRESHOLD = 80

    if score > THRESHOLD:
        return False

    return True


# ============================================================
# Label Management
# ============================================================

@callback(
    Output("labels-store", "data"),
    Output("label-input", "value"),

    Input(
        "add-label-btn",
        "n_clicks",
    ),

    State(
        "label-input",
        "value",
    ),

    State(
        "labels-store",
        "data",
    ),

    prevent_initial_call=True,
)
def add_label(
    n_clicks,
    label,
    labels,
):

    if not label:
        return labels, ""

    label = label.strip().lower()

    if label and label not in labels:

        if check(labels, label):
            labels.append(label)

    return labels, ""


# ============================================================
# Display Labels
# ============================================================

@callback(
    Output(
        "labels-container",
        "children",
    ),

    Input(
        "labels-store",
        "data",
    ),
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
# Probability Display
# ============================================================

def probability_table(probabilities):

    if not probabilities:
        return html.Div()

    rows = []

    for label, probability in probabilities.items():

        try:
            percentage = float(probability) * 100
        except Exception:
            percentage = 0

        rows.append(
            {
                "Classification": label,
                "Probability": round(
                    percentage,
                    2,
                ),
            }
        )

    rows.sort(
        key=lambda x: x["Probability"],
        reverse=True,
    )

    return dash_table.DataTable(

        data=rows,

        columns=[
            {
                "name": "Classification",
                "id": "Classification",
            },
            {
                "name": "Probability (%)",
                "id": "Probability",
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
            "padding": "10px",
            "textAlign": "left",
        },

        page_size=10,
    )


# ============================================================
# Single Classification Callback
# ============================================================

@callback(
    Output(
        "single-result",
        "children",
    ),

    Input(
        "classify-btn",
        "n_clicks",
    ),

    State(
        "single-prompt",
        "value",
    ),

    State(
        "labels-store",
        "data",
    ),

    prevent_initial_call=True,
)
def classify_single(
    n_clicks,
    prompt,
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
            labels=labels,
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        confidence = result.get(
            "confidence"
        )

        if confidence is not None:

            try:
                confidence_display = (
                    f"{float(confidence) * 100:.2f}%"
                )

            except Exception:
                confidence_display = str(
                    confidence
                )

        else:
            confidence_display = "N/A"


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

                    dbc.Row(
                        [

                            dbc.Col(
                                [

                                    html.Span(
                                        "Confidence",
                                        className="text-secondary",
                                    ),

                                    html.H4(
                                        confidence_display,
                                        className="mt-2",
                                    ),

                                ],
                                md=4,
                            ),

                            dbc.Col(
                                [

                                    html.Span(
                                        "Response Time",
                                        className="text-secondary",
                                    ),

                                    html.H4(
                                        f"{elapsed:.3f}s",
                                        className="mt-2",
                                    ),

                                ],
                                md=4,
                            ),

                            dbc.Col(
                                [

                                    html.Span(
                                        "Model",
                                        className="text-secondary",
                                    ),

                                    html.H4(
                                        DEFAULT_MODEL,
                                        className="mt-2",
                                    ),

                                ],
                                md=4,
                            ),

                        ],
                        className="mb-4",
                    ),

                    html.H5(
                        "Class Probabilities",
                        className="mb-3",
                    ),

                    probability_table(
                        result.get(
                            "probabilities",
                            {},
                        )
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
# Bulk Classification Callback
# ============================================================

@callback(
    Output(
        "bulk-results",
        "children",
    ),

    Output(
        "bulk-count",
        "children",
    ),

    Input(
        "bulk-classify-btn",
        "n_clicks",
    ),

    State(
        "bulk-prompts",
        "value",
    ),

    State(
        "labels-store",
        "data",
    ),

    prevent_initial_call=True,
)
def classify_bulk(
    n_clicks,
    prompts,
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
            labels=labels,
        )

        total_time = (
            time.perf_counter()
            - start_time
        )

        table_data = []

        for index, result in enumerate(
            results,
            start=1,
        ):

            confidence = result.get(
                "confidence"
            )

            if confidence is not None:

                try:
                    confidence = round(
                        float(confidence) * 100,
                        2,
                    )

                except Exception:
                    pass

            table_data.append(
                {
                    "No.": index,
                    "Request": result["request"],
                    "Classification": result["response"],
                    "Confidence (%)": confidence,
                    "Time (s)": result["time"],
                }
            )


        # ----------------------------------------------------
        # Results Table
        # ----------------------------------------------------

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
                    "name": "Confidence (%)",
                    "id": "Confidence (%)",
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
                },

                {
                    "if": {
                        "column_id": "Confidence (%)"
                    },

                    "fontWeight": "bold",
                },

            ],

            page_size=10,
        )


        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

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
                                    f"{total_time:.3f}s"
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
                                    f"{total_time / len(results):.3f}s"
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
                                    "Model",
                                    className="text-secondary",
                                ),

                                html.H3(
                                    "Jev"
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
        host="localhost",
        port=8050,
    )

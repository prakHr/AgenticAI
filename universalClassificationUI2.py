import time
from collections import Counter

from api_key import api_key
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

TYPESAFE_API_KEY = api_key

if not TYPESAFE_API_KEY:
    raise ValueError(
        "TYPESAFE_API_KEY not found. "
        "Add it to your .env file."
    )


client = TypeSafeClient(
    api_key=TYPESAFE_API_KEY
)


DEFAULT_MODEL = "jev-latest"

# UI/business threshold.
# This is NOT a Jev-defined confidence threshold.
REVIEW_THRESHOLD = 0.70

HIGH_CONFIDENCE_THRESHOLD = 0.90


# ============================================================
# Label Utilities
# ============================================================

def normalize_labels(labels):
    """
    Clean and normalize user-defined classification labels.
    """

    normalized = []

    for label in labels or []:

        label = str(label).strip().lower()

        if label and label not in normalized:
            normalized.append(label)

    return normalized


def prepare_labels(labels):
    """
    Normalize labels and ensure 'other' exists.
    """

    labels = normalize_labels(labels)

    if not labels:
        raise ValueError(
            "At least one classification label is required."
        )

    if "other" not in labels:
        labels.append("other")

    return labels


# ============================================================
# Probability Utilities
# ============================================================

def get_probability_details(probabilities):
    """
    Return the top classification, runner-up and probability margin.
    """

    if not probabilities:
        return {
            "top_label": None,
            "top_probability": 0.0,
            "runner_up": None,
            "runner_up_probability": 0.0,
            "margin": 0.0,
        }

    cleaned = {}

    for label, probability in probabilities.items():

        try:
            cleaned[label] = float(probability)

        except Exception:
            cleaned[label] = 0.0

    ranked = sorted(
        cleaned.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    top_label, top_probability = ranked[0]

    if len(ranked) > 1:

        runner_up, runner_up_probability = ranked[1]

    else:

        runner_up = None
        runner_up_probability = 0.0

    margin = (
        top_probability
        - runner_up_probability
    )

    return {
        "top_label": top_label,
        "top_probability": top_probability,
        "runner_up": runner_up,
        "runner_up_probability": runner_up_probability,
        "margin": margin,
    }


def get_confidence_status(confidence):
    """
    Convert confidence into a UI status.

    These thresholds are application-level thresholds.
    """

    try:
        confidence = float(confidence)

    except Exception:
        return {
            "label": "Unknown",
            "color": "secondary",
        }

    if confidence >= HIGH_CONFIDENCE_THRESHOLD:

        return {
            "label": "High confidence",
            "color": "success",
        }

    if confidence >= REVIEW_THRESHOLD:

        return {
            "label": "Medium confidence",
            "color": "warning",
        }

    return {
        "label": "Low confidence",
        "color": "danger",
    }


def needs_review(confidence, margin):
    """
    Determine whether the result should be reviewed.

    A result needs review if:
    - confidence is below REVIEW_THRESHOLD, OR
    - the top two classes are very close.

    The margin threshold is intentionally conservative.
    """

    try:
        confidence = float(confidence)
        margin = float(margin)

    except Exception:
        return True

    return (
        confidence < REVIEW_THRESHOLD
        or margin < 0.10
    )


# ============================================================
# Jev Classification
# ============================================================

def run_classification_execution(
    user_prompt: str,
    labels: list,
):
    """
    Classify one request using Jev Choice.
    """

    labels = prepare_labels(labels)

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

    classification = response.answers[
        "classification"
    ]

    # --------------------------------------------------------
    # ChoiceAnswer fields
    #
    # classification.choice
    # classification.confidence
    # classification.probabilities
    # --------------------------------------------------------

    label = classification.choice

    confidence = classification.confidence

    probabilities = (
        classification.probabilities
    )

    probability_details = (
        get_probability_details(
            probabilities
        )
    )

    review = needs_review(
        confidence,
        probability_details["margin"],
    )

    return {
        "request": user_prompt,
        "response": label,
        "confidence": confidence,
        "probabilities": probabilities,

        "runner_up": (
            probability_details["runner_up"]
        ),

        "runner_up_probability": (
            probability_details[
                "runner_up_probability"
            ]
        ),

        "margin": (
            probability_details["margin"]
        ),

        "needs_review": review,
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

    Each request is currently sent as
    one Jev decision.
    """

    results = []

    for prompt in user_prompts:

        start_time = time.perf_counter()

        result = run_classification_execution(
            user_prompt=prompt,
            labels=labels,
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        result["time"] = round(
            elapsed,
            3,
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


STAT_CARD_STYLE = {
    "backgroundColor": "#151515",
    "border": "1px solid #333",
    "borderRadius": "10px",
}


# ============================================================
# Layout
# ============================================================

app.layout = dbc.Container(
    [

        # ====================================================
        # Header
        # ====================================================

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


        # ====================================================
        # Configuration
        # ====================================================

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

                        html.Hr(),

                        html.Small(
                            [
                                "Review threshold: ",
                                html.Strong(
                                    "70%"
                                ),
                                " • Margin threshold: ",
                                html.Strong(
                                    "10%"
                                ),
                            ],
                            className="text-secondary",
                        ),

                    ],
                    style=CARD_STYLE,
                    color="dark",
                ),
                width=12,
            ),
        ),


        # ====================================================
        # Single Classification
        # ====================================================

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


        # ====================================================
        # Bulk Classification
        # ====================================================

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


        # ====================================================
        # Footer
        # ====================================================

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


        # ====================================================
        # Hidden Store
        # ====================================================

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
    """
    Prevent labels that are too similar to an existing label.
    """

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
    Output(
        "labels-store",
        "data",
    ),

    Output(
        "label-input",
        "value",
    ),

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

    labels = list(labels or [])

    if not label:
        return labels, ""

    label = label.strip().lower()

    if not label:
        return labels, ""

    if label in labels:
        return labels, ""

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

    if "other" not in labels:

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

    for label, probability in (
        probabilities.items()
    ):

        try:
            percentage = (
                float(probability)
                * 100
            )

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

        style_data_conditional=[

            {
                "if": {
                    "column_id": "Classification",
                    "filter_query": (
                        "{Classification} = "
                        f"'{rows[0]['Classification']}'"
                    ),
                },
                "fontWeight": "bold",
            },

        ],

        page_size=10,
    )


# ============================================================
# Probability Bars
# ============================================================

def probability_bars(probabilities):

    if not probabilities:
        return html.Div()

    sorted_probabilities = sorted(
        probabilities.items(),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    components = []

    for label, probability in (
        sorted_probabilities
    ):

        try:
            percentage = (
                float(probability)
                * 100
            )

        except Exception:
            percentage = 0

        components.append(

            html.Div(
                [

                    html.Div(
                        [

                            html.Span(
                                label,
                                className="fw-bold",
                            ),

                            html.Span(
                                f"{percentage:.2f}%",
                                className=(
                                    "text-secondary"
                                ),
                            ),

                        ],
                        className=(
                            "d-flex "
                            "justify-content-between "
                            "mb-1"
                        ),
                    ),

                    dbc.Progress(
                        value=percentage,
                        max=100,
                        striped=False,
                        animated=False,
                        style={
                            "height": "10px",
                            "borderRadius": "5px",
                        },
                    ),

                ],
                className="mb-3",
            )

        )

    return html.Div(components)


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

        try:
            confidence_float = float(
                confidence
            )

            confidence_display = (
                f"{confidence_float * 100:.2f}%"
            )

        except Exception:

            confidence_float = 0.0
            confidence_display = "N/A"


        runner_up = result.get(
            "runner_up"
        )

        runner_up_probability = (
            result.get(
                "runner_up_probability",
                0,
            )
        )

        try:
            runner_up_display = (
                f"{runner_up} "
                f"({float(runner_up_probability) * 100:.2f}%)"
            )

        except Exception:

            runner_up_display = (
                str(runner_up)
                if runner_up
                else "N/A"
            )


        margin = result.get(
            "margin",
            0,
        )

        try:
            margin_display = (
                f"{float(margin) * 100:.2f}%"
            )

        except Exception:

            margin_display = "N/A"


        status = get_confidence_status(
            confidence
        )

        review = result.get(
            "needs_review",
            True,
        )


        # ----------------------------------------------------
        # Review Badge
        # ----------------------------------------------------

        if review:

            review_badge = dbc.Badge(
                "Needs Review",
                color="warning",
                className="p-2",
            )

        else:

            review_badge = dbc.Badge(
                "Auto Accept",
                color="success",
                className="p-2",
            )


        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        return dbc.Card(

            dbc.CardBody(
                [

                    # ========================================
                    # Main Result
                    # ========================================

                    html.Div(
                        [

                            html.Div(
                                [

                                    html.Span(
                                        "Classification",
                                        className=(
                                            "text-secondary"
                                        ),
                                    ),

                                    html.H2(
                                        result["response"],
                                        className="mt-2",
                                    ),

                                ]
                            ),

                            html.Div(
                                [
                                    dbc.Badge(
                                        status["label"],
                                        color=status["color"],
                                        className="p-2 me-2",
                                    ),

                                    review_badge,
                                ],
                                className="mt-3",
                            ),

                        ]
                    ),

                    html.Hr(
                        className="my-4"
                    ),


                    # ========================================
                    # Metrics
                    # ========================================

                    dbc.Row(
                        [

                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody(
                                        [

                                            html.Small(
                                                "Confidence",
                                                className=(
                                                    "text-secondary"
                                                ),
                                            ),

                                            html.H4(
                                                confidence_display,
                                                className="mt-2",
                                            ),

                                        ]
                                    ),
                                    style=STAT_CARD_STYLE,
                                ),
                                md=3,
                            ),


                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody(
                                        [

                                            html.Small(
                                                "Runner-up",
                                                className=(
                                                    "text-secondary"
                                                ),
                                            ),

                                            html.H5(
                                                runner_up_display,
                                                className="mt-2",
                                            ),

                                        ]
                                    ),
                                    style=STAT_CARD_STYLE,
                                ),
                                md=3,
                            ),


                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody(
                                        [

                                            html.Small(
                                                "Confidence Margin",
                                                className=(
                                                    "text-secondary"
                                                ),
                                            ),

                                            html.H4(
                                                margin_display,
                                                className="mt-2",
                                            ),

                                        ]
                                    ),
                                    style=STAT_CARD_STYLE,
                                ),
                                md=3,
                            ),


                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody(
                                        [

                                            html.Small(
                                                "Response Time",
                                                className=(
                                                    "text-secondary"
                                                ),
                                            ),

                                            html.H4(
                                                f"{elapsed:.3f}s",
                                                className="mt-2",
                                            ),

                                        ]
                                    ),
                                    style=STAT_CARD_STYLE,
                                ),
                                md=3,
                            ),

                        ],
                        className="mb-4",
                    ),


                    # ========================================
                    # Model
                    # ========================================

                    html.Div(
                        [

                            html.Span(
                                "Model: ",
                                className="text-secondary",
                            ),

                            html.Strong(
                                DEFAULT_MODEL
                            ),

                        ],
                        className="mb-4",
                    ),


                    # ========================================
                    # Probability Bars
                    # ========================================

                    html.H5(
                        "Probability Distribution",
                        className="mb-3",
                    ),

                    probability_bars(
                        result.get(
                            "probabilities",
                            {},
                        )
                    ),


                    # ========================================
                    # Probability Table
                    # ========================================

                    html.H5(
                        "Detailed Probabilities",
                        className="mt-4 mb-3",
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

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        start_time = time.perf_counter()

        results = classify_multiple(
            user_prompts=user_prompts,
            labels=labels,
        )

        total_time = (
            time.perf_counter()
            - start_time
        )


        # ----------------------------------------------------
        # Aggregate Metrics
        # ----------------------------------------------------

        number_of_results = len(
            results
        )

        avg_time = (
            total_time / number_of_results
            if number_of_results
            else 0
        )

        requests_per_second = (
            number_of_results / total_time
            if total_time > 0
            else 0
        )


        high_confidence_count = 0

        review_count = 0

        for result in results:

            try:

                confidence = float(
                    result.get(
                        "confidence",
                        0,
                    )
                )

            except Exception:

                confidence = 0


            if (
                confidence
                >= HIGH_CONFIDENCE_THRESHOLD
            ):

                high_confidence_count += 1


            if result.get(
                "needs_review",
                True,
            ):

                review_count += 1


        # ----------------------------------------------------
        # Classification Distribution
        # ----------------------------------------------------

        distribution = Counter(
            result["response"]
            for result in results
        )


        # ----------------------------------------------------
        # Results Table
        # ----------------------------------------------------

        table_data = []


        for index, result in enumerate(
            results,
            start=1,
        ):

            confidence = result.get(
                "confidence"
            )

            try:

                confidence = round(
                    float(confidence) * 100,
                    2,
                )

            except Exception:

                confidence = None


            runner_up = result.get(
                "runner_up"
            )


            runner_up_probability = (
                result.get(
                    "runner_up_probability",
                    0,
                )
            )


            try:

                runner_up_display = (
                    f"{runner_up} "
                    f"({float(runner_up_probability) * 100:.2f}%)"
                )

            except Exception:

                runner_up_display = (
                    str(runner_up)
                    if runner_up
                    else "N/A"
                )


            margin = result.get(
                "margin",
                0,
            )


            try:

                margin = round(
                    float(margin) * 100,
                    2,
                )

            except Exception:

                margin = None


            review = (
                "Needs Review"
                if result.get(
                    "needs_review",
                    True,
                )
                else "Auto Accept"
            )


            table_data.append(
                {
                    "No.": index,

                    "Request": result[
                        "request"
                    ],

                    "Classification": result[
                        "response"
                    ],

                    "Confidence (%)": confidence,

                    "Runner-up": runner_up_display,

                    "Margin (%)": margin,

                    "Review": review,

                    "Time (s)": result[
                        "time"
                    ],
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
                    "name": "Runner-up",
                    "id": "Runner-up",
                },

                {
                    "name": "Margin (%)",
                    "id": "Margin (%)",
                },

                {
                    "name": "Review",
                    "id": "Review",
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

                {
                    "if": {
                        "filter_query": (
                            "{Review} = "
                            "'Needs Review'"
                        ),
                    },

                    "backgroundColor": (
                        "#332b00"
                    ),
                },

            ],

            page_size=10,

        )


        # ----------------------------------------------------
        # Distribution Table
        # ----------------------------------------------------

        distribution_data = []

        for label, count in (
            distribution.most_common()
        ):

            percentage = (
                count
                / number_of_results
                * 100
            )

            distribution_data.append(
                {
                    "Classification": label,
                    "Count": count,
                    "Percentage (%)": round(
                        percentage,
                        2,
                    ),
                }
            )


        distribution_table = (
            dash_table.DataTable(

                data=distribution_data,

                columns=[

                    {
                        "name": "Classification",
                        "id": "Classification",
                    },

                    {
                        "name": "Count",
                        "id": "Count",
                    },

                    {
                        "name": "Percentage (%)",
                        "id": "Percentage (%)",
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
        )


        # ----------------------------------------------------
        # Summary Cards
        # ----------------------------------------------------

        summary = dbc.Row(
            [

                # Requests
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [

                                html.Small(
                                    "Requests",
                                    className=(
                                        "text-secondary"
                                    ),
                                ),

                                html.H3(
                                    number_of_results
                                ),

                            ]
                        ),
                        style=STAT_CARD_STYLE,
                    ),
                    md=3,
                    className="mb-3",
                ),


                # Total Time
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [

                                html.Small(
                                    "Total Time",
                                    className=(
                                        "text-secondary"
                                    ),
                                ),

                                html.H3(
                                    f"{total_time:.3f}s"
                                ),

                            ]
                        ),
                        style=STAT_CARD_STYLE,
                    ),
                    md=3,
                    className="mb-3",
                ),


                # Average Time
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [

                                html.Small(
                                    "Avg Time",
                                    className=(
                                        "text-secondary"
                                    ),
                                ),

                                html.H3(
                                    f"{avg_time:.3f}s"
                                ),

                            ]
                        ),
                        style=STAT_CARD_STYLE,
                    ),
                    md=3,
                    className="mb-3",
                ),


                # Throughput
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [

                                html.Small(
                                    "Requests / Sec",
                                    className=(
                                        "text-secondary"
                                    ),
                                ),

                                html.H3(
                                    f"{requests_per_second:.2f}"
                                ),

                            ]
                        ),
                        style=STAT_CARD_STYLE,
                    ),
                    md=3,
                    className="mb-3",
                ),

            ],
            className="mb-2",
        )


        # ----------------------------------------------------
        # Quality Summary
        # ----------------------------------------------------

        quality_summary = dbc.Row(
            [

                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [

                                html.Small(
                                    "High Confidence",
                                    className=(
                                        "text-secondary"
                                    ),
                                ),

                                html.H3(
                                    high_confidence_count
                                ),

                                html.Small(
                                    (
                                        f"{high_confidence_count / number_of_results * 100:.1f}%"
                                        if number_of_results
                                        else "0%"
                                    ),
                                    className=(
                                        "text-success"
                                    ),
                                ),

                            ]
                        ),
                        style=STAT_CARD_STYLE,
                    ),
                    md=4,
                    className="mb-3",
                ),


                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [

                                html.Small(
                                    "Needs Review",
                                    className=(
                                        "text-secondary"
                                    ),
                                ),

                                html.H3(
                                    review_count
                                ),

                                html.Small(
                                    (
                                        f"{review_count / number_of_results * 100:.1f}%"
                                        if number_of_results
                                        else "0%"
                                    ),
                                    className=(
                                        "text-warning"
                                    ),
                                ),

                            ]
                        ),
                        style=STAT_CARD_STYLE,
                    ),
                    md=4,
                    className="mb-3",
                ),


                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [

                                html.Small(
                                    "Model",
                                    className=(
                                        "text-secondary"
                                    ),
                                ),

                                html.H3(
                                    "Jev"
                                ),

                                html.Small(
                                    DEFAULT_MODEL,
                                    className=(
                                        "text-secondary"
                                    ),
                                ),

                            ]
                        ),
                        style=STAT_CARD_STYLE,
                    ),
                    md=4,
                    className="mb-3",
                ),

            ],
            className="mb-4",
        )


        # ----------------------------------------------------
        # Distribution
        # ----------------------------------------------------

        distribution_section = dbc.Card(
            dbc.CardBody(
                [

                    html.H5(
                        "Classification Distribution",
                        className="mb-3",
                    ),

                    distribution_table,

                ]
            ),
            style=STAT_CARD_STYLE,
            className="mb-4",
        )


        # ----------------------------------------------------
        # Final Result
        # ----------------------------------------------------

        return (

            html.Div(
                [

                    summary,

                    quality_summary,

                    distribution_section,

                    html.H5(
                        "Classification Results",
                        className="mb-3",
                    ),

                    table,

                ]
            ),

            (
                f"{number_of_results} "
                f"requests classified • "
                f"{review_count} need review"
            ),

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

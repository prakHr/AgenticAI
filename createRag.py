import os

# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()


# ============================================================
# IMPORTS
# ============================================================

import hashlib
import multiprocessing
import re

import numpy as np

from dotenv import load_dotenv

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from mpire import WorkerPool

from transformers import (
    pipeline,
    AutoTokenizer
)

from qdrant_client import QdrantClient

from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
)

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from rank_bm25 import BM25Okapi


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

QDRANT_URL = os.getenv(
    "QDRANT_URL"
)

QDRANT_API_KEY = os.getenv(
    "QDRANT_API_KEY"
)

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)


# ============================================================
# CONFIGURATION
# ============================================================

BART_MODEL = (
    "facebook/bart-large-cnn"
)

EMBEDDING_MODEL = (
    "all-MiniLM-L6-v2"
)

RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

EMBEDDING_SIZE = 384


# ============================================================
# RECURSIVE CHUNKING
# ============================================================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# ============================================================
# MULTIPROCESSING
# ============================================================

NUM_WORKERS = max(
    min(
        multiprocessing.cpu_count() // 2,
        2
    ),
    1
)


# ============================================================
# HYBRID SEARCH CONFIGURATION
# ============================================================

# Number of candidates retrieved from Qdrant
VECTOR_CANDIDATES = 50

# Number of candidates retrieved from BM25
BM25_CANDIDATES = 50

# Number of candidates sent to CrossEncoder
HYBRID_CANDIDATES = 50

# Final number of results
# top_k is passed to create_rag_pipeline()


# ============================================================
# RRF CONFIGURATION
# ============================================================

RRF_K = 60

VECTOR_WEIGHT = 0.5

BM25_WEIGHT = 0.5


# ============================================================
# NEAR-DUPLICATE CONFIGURATION
# ============================================================

# Cosine similarity above this value means that two
# chunks are considered too similar.
#
# 0.90 -> aggressive duplicate suppression
# 0.95 -> moderate
# 0.98 -> only almost-identical chunks
#
# Recommended starting point:
SEMANTIC_DUPLICATE_THRESHOLD = 0.92


# ============================================================
# GET LOADER
# ============================================================

def get_loader(file_path):

    """
    Loads PDF, TXT, DOCX and DOC files.

    PDF:
        - Extracts all pages.
        - Adds [PAGE X] markers.
        - Combines pages into one continuous document.

    This allows recursive splitting across page boundaries.
    """

    ext = os.path.splitext(
        file_path
    )[1].lower()

    try:

        # ====================================================
        # PDF
        # ====================================================

        if ext == ".pdf":

            pages = PyPDFLoader(
                file_path
            ).load()

            if not pages:
                return None

            complete_text = []

            for page_number, page in enumerate(
                pages,
                start=1
            ):

                page_text = (
                    page.page_content.strip()
                )

                if not page_text:
                    continue

                page_block = (
                    f"\n[PAGE {page_number}]\n"
                    f"{page_text}\n"
                )

                complete_text.append(
                    page_block
                )

            full_text = "\n".join(
                complete_text
            )

            if not full_text.strip():
                return None

            metadata = {}

            if pages[0].metadata:

                metadata.update(
                    pages[0].metadata
                )

            metadata["source"] = file_path

            metadata["total_pages"] = len(
                pages
            )

            metadata["file_type"] = "pdf"

            return [
                {
                    "page_content": full_text,
                    "metadata": metadata
                }
            ]


        # ====================================================
        # TXT
        # ====================================================

        elif ext == ".txt":

            documents = TextLoader(
                file_path,
                encoding="utf-8"
            ).load()

            results = []

            for document in documents:

                results.append(
                    {
                        "page_content":
                            document.page_content,

                        "metadata": {
                            **document.metadata,

                            "source":
                                file_path,

                            "file_type":
                                "txt"
                        }
                    }
                )

            return results


        # ====================================================
        # DOC / DOCX
        # ====================================================

        elif ext in [
            ".docx",
            ".doc"
        ]:

            documents = (
                UnstructuredWordDocumentLoader(
                    file_path
                ).load()
            )

            results = []

            for document in documents:

                results.append(
                    {
                        "page_content":
                            document.page_content,

                        "metadata": {

                            **document.metadata,

                            "source":
                                file_path,

                            "file_type":
                                ext[1:]
                        }
                    }
                )

            return results


        # ====================================================
        # UNSUPPORTED
        # ====================================================

        return None


    except Exception as e:

        print(
            f"\nError loading file:"
            f"\n{file_path}"
            f"\n{e}"
        )

        return None


# ============================================================
# GET FILES
# ============================================================

def get_files(folder_path):

    supported_extensions = {
        ".pdf",
        ".txt",
        ".docx",
        ".doc"
    }

    files = []

    for file_name in os.listdir(
        folder_path
    ):

        file_path = os.path.join(
            folder_path,
            file_name
        )

        if not os.path.isfile(
            file_path
        ):
            continue

        ext = os.path.splitext(
            file_name
        )[1].lower()

        if ext in supported_extensions:

            files.append(
                file_path
            )

    return files


# ============================================================
# LOAD BART MODEL
# ============================================================

def load_big_model(
    worker_state
):

    tokenizer = (
        AutoTokenizer.from_pretrained(
            BART_MODEL
        )
    )

    summarizer = pipeline(
        "summarization",

        model=BART_MODEL,

        tokenizer=tokenizer,

        device=-1
    )

    worker_state[
        "tokenizer"
    ] = tokenizer

    worker_state[
        "summarizer"
    ] = summarizer


# ============================================================
# CREATE TEXT SPLITTER
# ============================================================

def create_text_splitter():

    splitter = (
        RecursiveCharacterTextSplitter(

            chunk_size=CHUNK_SIZE,

            chunk_overlap=CHUNK_OVERLAP,

            separators=[
                "\n\n",
                "\n",
                ". ",
                "! ",
                "? ",
                "; ",
                ", ",
                " ",
                ""
            ],

            length_function=len,

            is_separator_regex=False
        )
    )

    return splitter


# ============================================================
# FIND PAGE NUMBERS
# ============================================================

def get_page_numbers(
    text
):

    matches = re.findall(
        r"\[PAGE\s+(\d+)\]",
        text
    )

    page_numbers = []

    for match in matches:

        page_number = int(
            match
        )

        if page_number not in page_numbers:

            page_numbers.append(
                page_number
            )

    return page_numbers


# ============================================================
# CLEAN PAGE MARKERS
# ============================================================

def clean_page_markers(
    text
):

    text = re.sub(
        r"\[PAGE\s+\d+\]",
        "",
        text
    )

    return text.strip()


# ============================================================
# CREATE STABLE CHUNK ID
# ============================================================

def create_chunk_id(
    text,
    file_path,
    chunk_index
):

    """
    Creates a stable ID for every chunk.
    """

    raw = (
        f"{file_path}|"
        f"{chunk_index}|"
        f"{text}"
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


# ============================================================
# SPLIT DOCUMENTS
# ============================================================

def split_documents(
    documents
):

    """
    Cross-page recursive chunking.
    """

    splitter = (
        create_text_splitter()
    )

    split_docs = []

    for document in documents:

        full_text = (
            document[
                "page_content"
            ]
        )

        metadata = (
            document[
                "metadata"
            ]
        )

        if not full_text.strip():
            continue

        chunks = (
            splitter.split_text(
                full_text
            )
        )

        for chunk_index, chunk in enumerate(
            chunks
        ):

            if not chunk.strip():
                continue

            page_numbers = (
                get_page_numbers(
                    chunk
                )
            )

            clean_text = (
                clean_page_markers(
                    chunk
                )
            )

            if not clean_text:
                continue

            chunk_id = (
                create_chunk_id(
                    clean_text,
                    metadata.get(
                        "source",
                        ""
                    ),
                    chunk_index
                )
            )

            chunk_metadata = {

                **metadata,

                "chunk_id":
                    chunk_id,

                "chunk_index":
                    chunk_index,

                "page_numbers":
                    page_numbers,

                "start_page":
                    (
                        min(page_numbers)
                        if page_numbers
                        else None
                    ),

                "end_page":
                    (
                        max(page_numbers)
                        if page_numbers
                        else None
                    )
            }

            split_docs.append(
                {
                    "page_content":
                        clean_text,

                    "metadata":
                        chunk_metadata
                }
            )

    return split_docs


# ============================================================
# EXACT CHUNK DEDUPLICATION
# ============================================================

def deduplicate_chunks(
    documents
):

    """
    Removes exact duplicate chunks.

    Whitespace differences and case differences
    are ignored.
    """

    seen = set()

    unique_documents = []

    duplicates = 0

    for document in documents:

        text = (
            document[
                "page_content"
            ].strip()
        )

        if not text:
            continue

        normalized_text = re.sub(
            r"\s+",
            " ",
            text
        ).strip().lower()

        if normalized_text in seen:

            duplicates += 1

            continue

        seen.add(
            normalized_text
        )

        unique_documents.append(
            document
        )

    print(
        f"\nExact duplicate chunks removed: "
        f"{duplicates}"
    )

    print(
        f"Unique chunks remaining: "
        f"{len(unique_documents)}"
    )

    return unique_documents


# ============================================================
# SUMMARIZE CHUNK
# ============================================================

def summarize(
    worker_state,
    page_content,
    metadata
):

    summarizer = (
        worker_state[
            "summarizer"
        ]
    )

    text = (
        page_content.strip()
    )

    if not text:

        return {
            "text": "",

            "original_text": "",

            "metadata": metadata
        }

    try:

        input_length = len(
            text.split()
        )

        # ====================================================
        # SMALL CHUNK
        # ====================================================

        if input_length < 30:

            return {
                "text": text,

                "original_text":
                    text,

                "metadata":
                    metadata
            }

        # ====================================================
        # SUMMARY LENGTH
        # ====================================================

        max_len = min(
            100,
            max(
                30,
                int(
                    input_length * 0.6
                )
            )
        )

        min_len = min(
            30,
            max(
                10,
                int(
                    max_len * 0.4
                )
            )
        )

        if min_len >= max_len:

            min_len = max(
                5,
                max_len // 2
            )

        # ====================================================
        # BART
        # ====================================================

        result = summarizer(

            text,

            max_length=max_len,

            min_length=min_len,

            do_sample=False,

            truncation=True
        )

        summary_text = (
            result[0][
                "summary_text"
            ]
        )

        return {

            "text":
                summary_text,

            "original_text":
                text,

            "metadata":
                metadata
        }

    except Exception:

        return {

            "text":
                text,

            "original_text":
                text,

            "metadata":
                metadata
        }


# ============================================================
# MODIFY DOCUMENTS
# ============================================================

def modify_docs(
    documents
):

    results = []

    metadata_set = set()

    for document in documents:

        metadata = (
            document[
                "metadata"
            ]
        )

        result = {

            "text":
                document[
                    "text"
                ],

            "original_text":
                document[
                    "original_text"
                ]
        }

        for key, value in (
            metadata.items()
        ):

            if key == "source":

                file_path = value

                ext = (
                    os.path.splitext(
                        file_path
                    )[1]
                    .lower()[1:]
                )

                result[
                    "source"
                ] = ext

                result[
                    "file_path"
                ] = file_path

                metadata_set.add(
                    (
                        "source",
                        ext
                    )
                )

            else:

                result[key] = value

        results.append(
            result
        )

    return (
        results,
        metadata_set
    )


# ============================================================
# LOAD + SPLIT + DEDUPLICATE + SUMMARIZE
# ============================================================

def get_list_of_dicts(
    folder_path,
    progress_bar
):

    # ========================================================
    # GET FILES
    # ========================================================

    files = get_files(
        folder_path
    )

    print(
        f"\nFound {len(files)} files."
    )

    # ========================================================
    # LOAD FILES
    # ========================================================

    with WorkerPool(
        n_jobs=NUM_WORKERS,
        daemon=False
    ) as pool:

        loaded_results = (
            pool.map(
                get_loader,
                files,
                progress_bar=
                    progress_bar
            )
        )

    # ========================================================
    # FLATTEN
    # ========================================================

    documents = []

    for result in loaded_results:

        if result is None:
            continue

        documents.extend(
            result
        )

    print(
        f"Loaded {len(documents)} "
        f"complete documents."
    )

    # ========================================================
    # CROSS-PAGE RECURSIVE SPLITTING
    # ========================================================

    split_docs = (
        split_documents(
            documents
        )
    )

    print(
        f"Created {len(split_docs)} "
        f"cross-page chunks."
    )

    # ========================================================
    # EXACT DEDUPLICATION
    # ========================================================

    split_docs = (
        deduplicate_chunks(
            split_docs
        )
    )

    # ========================================================
    # SUMMARIZATION
    # ========================================================

    with WorkerPool(
        n_jobs=NUM_WORKERS,
        daemon=False,
        use_worker_state=True
    ) as pool:

        summarized_docs = (
            pool.map(

                summarize,

                split_docs,

                progress_bar=
                    progress_bar,

                worker_init=
                    load_big_model
            )
        )

    print(
        f"Summarized "
        f"{len(summarized_docs)} chunks."
    )

    # ========================================================
    # MODIFY METADATA
    # ========================================================

    results, metadata_set = (
        modify_docs(
            summarized_docs
        )
    )

    return (
        results,
        metadata_set
    )


# ============================================================
# TOKENIZE FOR BM25
# ============================================================

def tokenize_text(
    text
):

    """
    Tokenizer for BM25.

    Lowercase + word tokenization.
    """

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


# ============================================================
# BUILD BM25
# ============================================================

def build_bm25_index(
    documents
):

    print(
        "\nBuilding BM25 index..."
    )

    tokenized_documents = []

    for document in documents:

        text = document[
            "original_text"
        ]

        tokens = (
            tokenize_text(
                text
            )
        )

        tokenized_documents.append(
            tokens
        )

    bm25 = BM25Okapi(
        tokenized_documents
    )

    print(
        f"BM25 index created for "
        f"{len(documents)} chunks."
    )

    return bm25


# ============================================================
# BM25 SEARCH
# ============================================================

def bm25_search(
    bm25,
    query,
    top_k
):

    query_tokens = (
        tokenize_text(
            query
        )
    )

    if not query_tokens:

        return []

    scores = (
        bm25.get_scores(
            query_tokens
        )
    )

    ranked_indices = (
        np.argsort(
            scores
        )[::-1]
    )

    ranked_indices = (
        ranked_indices[
            :top_k
        ]
    )

    results = []

    for index in ranked_indices:

        results.append(
            {
                "index":
                    int(index),

                "bm25_score":
                    float(
                        scores[index]
                    )
            }
        )

    return results


# ============================================================
# RRF
# ============================================================

def reciprocal_rank_fusion(
    vector_results,
    bm25_results,
    k=60,
    vector_weight=0.5,
    bm25_weight=0.5
):

    """
    Weighted Reciprocal Rank Fusion.

    Each document can occur only once in
    the fused result.
    """

    scores = {}

    metadata = {}

    # ========================================================
    # VECTOR
    # ========================================================

    for rank, result in enumerate(
        vector_results,
        start=1
    ):

        doc_index = (
            result[
                "index"
            ]
        )

        rrf_score = (
            vector_weight /
            (k + rank)
        )

        scores[
            doc_index
        ] = (
            scores.get(
                doc_index,
                0.0
            )
            +
            rrf_score
        )

        metadata.setdefault(
            doc_index,
            {}
        )

        metadata[
            doc_index
        ][
            "vector_rank"
        ] = rank

        metadata[
            doc_index
        ][
            "vector_score"
        ] = result[
            "qdrant_score"
        ]

    # ========================================================
    # BM25
    # ========================================================

    for rank, result in enumerate(
        bm25_results,
        start=1
    ):

        doc_index = (
            result[
                "index"
            ]
        )

        rrf_score = (
            bm25_weight /
            (k + rank)
        )

        scores[
            doc_index
        ] = (
            scores.get(
                doc_index,
                0.0
            )
            +
            rrf_score
        )

        metadata.setdefault(
            doc_index,
            {}
        )

        metadata[
            doc_index
        ][
            "bm25_rank"
        ] = rank

        metadata[
            doc_index
        ][
            "bm25_score"
        ] = result[
            "bm25_score"
        ]

    # ========================================================
    # SORT
    # ========================================================

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # ========================================================
    # CREATE RESULT
    # ========================================================

    results = []

    for doc_index, rrf_score in ranked:

        result_metadata = (
            metadata[
                doc_index
            ]
        )

        results.append(
            {

                "index":
                    doc_index,

                "rrf_score":
                    float(
                        rrf_score
                    ),

                "vector_rank":
                    result_metadata.get(
                        "vector_rank"
                    ),

                "bm25_rank":
                    result_metadata.get(
                        "bm25_rank"
                    ),

                "vector_score":
                    result_metadata.get(
                        "vector_score"
                    ),

                "bm25_score":
                    result_metadata.get(
                        "bm25_score"
                    )
            }
        )

    return results


# ============================================================
# LOAD CROSS ENCODER
# ============================================================

def load_cross_encoder():

    print(
        "\nLoading CrossEncoder..."
    )

    reranker = CrossEncoder(
        RERANKER_MODEL
    )

    print(
        "CrossEncoder loaded."
    )

    return reranker


# ============================================================
# CROSS ENCODER RERANK
# ============================================================

def cross_encoder_rerank(
    query,
    hybrid_results,
    documents,
    reranker
):

    if not hybrid_results:

        return []

    # ========================================================
    # CREATE QUERY-DOCUMENT PAIRS
    # ========================================================

    pairs = []

    for result in hybrid_results:

        index = (
            result[
                "index"
            ]
        )

        document_text = (
            documents[
                index
            ][
                "original_text"
            ]
        )

        pairs.append(
            (
                query,
                document_text
            )
        )

    # ========================================================
    # PREDICT
    # ========================================================

    scores = (
        reranker.predict(
            pairs,
            show_progress_bar=True
        )
    )

    # ========================================================
    # ATTACH SCORES
    # ========================================================

    reranked_results = []

    for result, score in zip(
        hybrid_results,
        scores
    ):

        index = (
            result[
                "index"
            ]
        )

        reranked_results.append(
            {

                **result,

                "cross_encoder_score":
                    float(score),

                "payload":
                    documents[
                        index
                    ]
            }
        )

    # ========================================================
    # SORT
    # ========================================================

    reranked_results.sort(
        key=lambda x:
            x[
                "cross_encoder_score"
            ],
        reverse=True
    )

    return reranked_results


# ============================================================
# SEMANTIC DUPLICATE SUPPRESSION
# ============================================================

def remove_semantic_duplicates(
    results,
    embedding_model,
    top_k,
    threshold=0.92
):

    """
    Removes highly similar chunks from the final result.

    This handles cases where:

        Chunk A = first 1000 chars
        Chunk B = overlapping 150 chars

    and both are returned by the retriever.

    The CrossEncoder score is considered first.
    Therefore, the better-ranked chunk is preserved.
    """

    if not results:

        return []

    selected = []

    selected_embeddings = []

    texts = [
        result[
            "payload"
        ][
            "original_text"
        ]

        for result in results
    ]

    # ========================================================
    # EMBEDDINGS
    # ========================================================

    embeddings = (
        embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )
    )

    # ========================================================
    # GREEDY DEDUPLICATION
    # ========================================================

    for index, result in enumerate(
        results
    ):

        current_embedding = (
            embeddings[index]
        )

        is_duplicate = False

        for selected_embedding in (
            selected_embeddings
        ):

            similarity = float(
                np.dot(
                    current_embedding,
                    selected_embedding
                )
            )

            if similarity >= threshold:

                is_duplicate = True

                break

        if is_duplicate:

            continue

        selected.append(
            result
        )

        selected_embeddings.append(
            current_embedding
        )

        if len(selected) >= top_k:

            break

    return selected


# ============================================================
# CREATE RAG PIPELINE
# ============================================================

def create_rag_pipeline(
    folder_path,
    progress_bar,
    COLLECTION_NAME,
    EMBEDDING_SIZE,
    query,
    top_k
):

    # ========================================================
    # LOAD DOCUMENTS
    # ========================================================

    documents, metadata_set = (
        get_list_of_dicts(
            folder_path,
            progress_bar
        )
    )

    if not documents:

        print(
            "No documents found."
        )

        return []


    # ========================================================
    # QDRANT CLIENT
    # ========================================================

    client = QdrantClient(

        url=QDRANT_URL,

        api_key=QDRANT_API_KEY,

        timeout=60
    )


    # ========================================================
    # DELETE EXISTING COLLECTION
    # ========================================================

    if client.collection_exists(
        COLLECTION_NAME
    ):

        print(
            f"\nDeleting existing "
            f"collection: "
            f"{COLLECTION_NAME}"
        )

        client.delete_collection(
            COLLECTION_NAME
        )


    # ========================================================
    # CREATE COLLECTION
    # ========================================================

    client.create_collection(

        collection_name=
            COLLECTION_NAME,

        vectors_config=
            VectorParams(

                size=
                    EMBEDDING_SIZE,

                distance=
                    Distance.COSINE
            )
    )


    # ========================================================
    # PAYLOAD INDEX
    # ========================================================

    client.create_payload_index(

        collection_name=
            COLLECTION_NAME,

        field_name=
            "source",

        field_schema=
            PayloadSchemaType.KEYWORD
    )


    # ========================================================
    # EMBEDDING MODEL
    # ========================================================

    print(
        "\nLoading embedding model..."
    )

    embedding_model = (
        SentenceTransformer(
            EMBEDDING_MODEL
        )
    )


    # ========================================================
    # CREATE EMBEDDINGS
    # ========================================================

    original_texts = [

        document[
            "original_text"
        ]

        for document in documents
    ]

    print(
        f"\nCreating embeddings for "
        f"{len(original_texts)} chunks..."
    )

    embeddings = (
        embedding_model.encode(

            original_texts,

            show_progress_bar=True,

            normalize_embeddings=True
        )
    )


    # ========================================================
    # CREATE QDRANT POINTS
    # ========================================================

    points = []

    for index, document in enumerate(
        documents
    ):

        point = PointStruct(

            id=index + 1,

            vector=
                embeddings[
                    index
                ].tolist(),

            payload={

                "chunk_id":
                    document.get(
                        "chunk_id"
                    ),

                "text":
                    document[
                        "text"
                    ],

                "original_text":
                    document[
                        "original_text"
                    ],

                "file_path":
                    document.get(
                        "file_path"
                    ),

                "source":
                    document.get(
                        "source"
                    ),

                "chunk_index":
                    document.get(
                        "chunk_index"
                    ),

                "page_numbers":
                    document.get(
                        "page_numbers",
                        []
                    ),

                "start_page":
                    document.get(
                        "start_page"
                    ),

                "end_page":
                    document.get(
                        "end_page"
                    ),

                **{

                    key: value

                    for key, value
                    in document.items()

                    if key not in [

                        "text",

                        "original_text",

                        "file_path",

                        "source",

                        "chunk_id",

                        "chunk_index",

                        "page_numbers",

                        "start_page",

                        "end_page"
                    ]
                }
            }
        )

        points.append(
            point
        )


    # ========================================================
    # INSERT INTO QDRANT
    # ========================================================

    print(
        "\nUploading vectors to Qdrant..."
    )

    client.upsert(

        collection_name=
            COLLECTION_NAME,

        points=
            points
    )

    print(
        f"Inserted {len(points)} "
        f"points into Qdrant."
    )


    # ========================================================
    # BUILD BM25
    # ========================================================

    bm25 = (
        build_bm25_index(
            documents
        )
    )


    # ========================================================
    # LOAD CROSS ENCODER
    # ========================================================

    reranker = (
        load_cross_encoder()
    )


    # ========================================================
    # CREATE FILTER
    # ========================================================

    must_list = []

    for key, value in metadata_set:

        condition = FieldCondition(

            key=key,

            match=MatchValue(
                value=value
            )
        )

        must_list.append(
            condition
        )


    if must_list:

        filters = Filter(
            should=must_list
        )

    else:

        filters = None


    # ========================================================
    # VECTOR SEARCH
    # ========================================================

    def vector_search(
        query,
        query_filter,
        candidate_count
    ):

        query_vector = (
            embedding_model.encode(
                query,
                normalize_embeddings=True
            ).tolist()
        )

        results = (
            client.query_points(

                collection_name=
                    COLLECTION_NAME,

                query=
                    query_vector,

                limit=
                    candidate_count,

                with_payload=True,

                query_filter=
                    query_filter
            ).points
        )

        return results


    # ========================================================
    # STEP 1
    # QDRANT VECTOR SEARCH
    # ========================================================

    print("\n")
    print("=" * 100)
    print(
        "STEP 1 - QDRANT VECTOR SEARCH"
    )
    print("=" * 100)

    vector_results_raw = (
        vector_search(

            query,

            filters,

            VECTOR_CANDIDATES
        )
    )

    print(
        f"Qdrant returned "
        f"{len(vector_results_raw)} "
        f"candidates."
    )


    # ========================================================
    # CONVERT QDRANT RESULTS
    # ========================================================

    vector_results = []

    for result in (
        vector_results_raw
    ):

        document_index = (
            int(result.id) - 1
        )

        vector_results.append(
            {

                "index":
                    document_index,

                "qdrant_score":
                    float(
                        result.score
                    )
            }
        )


    # ========================================================
    # STEP 2
    # BM25
    # ========================================================

    print("\n")
    print("=" * 100)
    print(
        "STEP 2 - BM25 SEARCH"
    )
    print("=" * 100)

    bm25_results = (
        bm25_search(

            bm25,

            query,

            BM25_CANDIDATES
        )
    )

    print(
        f"BM25 returned "
        f"{len(bm25_results)} "
        f"candidates."
    )


    # ========================================================
    # STEP 3
    # RRF
    # ========================================================

    print("\n")
    print("=" * 100)
    print(
        "STEP 3 - RRF HYBRID FUSION"
    )
    print("=" * 100)

    hybrid_results = (
        reciprocal_rank_fusion(

            vector_results,

            bm25_results,

            k=RRF_K,

            vector_weight=
                VECTOR_WEIGHT,

            bm25_weight=
                BM25_WEIGHT
        )
    )

    print(
        f"RRF produced "
        f"{len(hybrid_results)} "
        f"unique candidates."
    )


    # ========================================================
    # LIMIT HYBRID CANDIDATES
    # ========================================================

    hybrid_results = (
        hybrid_results[
            :HYBRID_CANDIDATES
        ]
    )

    print(
        f"Passing "
        f"{len(hybrid_results)} "
        f"candidates to CrossEncoder."
    )


    # ========================================================
    # STEP 4
    # CROSS ENCODER
    # ========================================================

    print("\n")
    print("=" * 100)
    print(
        "STEP 4 - CROSS ENCODER RERANKING"
    )
    print("=" * 100)

    reranked_results = (
        cross_encoder_rerank(

            query,

            hybrid_results,

            documents,

            reranker
        )
    )


    # ========================================================
    # STEP 5
    # SEMANTIC DUPLICATE REMOVAL
    # ========================================================

    print("\n")
    print("=" * 100)
    print(
        "STEP 5 - SEMANTIC DUPLICATE SUPPRESSION"
    )
    print("=" * 100)

    final_results = (
        remove_semantic_duplicates(

            reranked_results,

            embedding_model,

            top_k,

            threshold=
                SEMANTIC_DUPLICATE_THRESHOLD
        )
    )

    print(
        f"Final unique results: "
        f"{len(final_results)}"
    )


    # ========================================================
    # RETURN
    # ========================================================

    return final_results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    folder_path = (
        r"C:\Users\gprak\Downloads"
    )

    progress_bar = True

    COLLECTION_NAME = (
        "knowledge_filter"
    )

    EMBEDDING_SIZE = 384

    query = (
        "Give details about python."
    )

    top_k = 3


    # ========================================================
    # RUN RAG
    # ========================================================

    results = (
        create_rag_pipeline(

            folder_path,

            progress_bar,

            COLLECTION_NAME,

            EMBEDDING_SIZE,

            query,

            top_k
        )
    )


    # ========================================================
    # PRINT FINAL RESULTS
    # ========================================================

    print("\n")

    print("=" * 100)

    print(
        "FINAL HYBRID SEARCH RESULTS"
    )

    print("=" * 100)


    for index, result in enumerate(
        results,
        start=1
    ):

        payload = (
            result[
                "payload"
            ]
        )

        print("\n")

        print("-" * 100)

        print(
            f"RESULT {index}"
        )

        print("-" * 100)


        # ====================================================
        # RANKING INFORMATION
        # ====================================================

        print(
            "\nChunk ID:",
            payload.get(
                "chunk_id"
            )
        )

        print(
            "Final CrossEncoder Score:",
            result.get(
                "cross_encoder_score"
            )
        )

        print(
            "RRF Score:",
            result.get(
                "rrf_score"
            )
        )

        print(
            "Qdrant Score:",
            result.get(
                "vector_score"
            )
        )

        print(
            "BM25 Score:",
            result.get(
                "bm25_score"
            )
        )

        print(
            "Vector Rank:",
            result.get(
                "vector_rank"
            )
        )

        print(
            "BM25 Rank:",
            result.get(
                "bm25_rank"
            )
        )


        # ====================================================
        # DOCUMENT INFORMATION
        # ====================================================

        print(
            "\nFile:",
            payload.get(
                "file_path"
            )
        )

        print(
            "Pages:",
            payload.get(
                "page_numbers"
            )
        )

        print(
            "Start Page:",
            payload.get(
                "start_page"
            )
        )

        print(
            "End Page:",
            payload.get(
                "end_page"
            )
        )

        print(
            "Chunk:",
            payload.get(
                "chunk_index"
            )
        )


        # ====================================================
        # SUMMARY
        # ====================================================

        print(
            "\nSUMMARY:"
        )

        print(
            payload.get(
                "text"
            )
        )


        # ====================================================
        # ORIGINAL
        # ====================================================

        print(
            "\nORIGINAL CHUNK:"
        )

        print(
            payload.get(
                "original_text"
            )
        )

        print(
            "\n" + "=" * 100
        )

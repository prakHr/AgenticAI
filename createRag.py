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

import multiprocessing

from dotenv import load_dotenv

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from mpire import WorkerPool

from transformers import pipeline, AutoTokenizer

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

from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ============================================================
# CONFIGURATION
# ============================================================

BART_MODEL = "facebook/bart-large-cnn"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

EMBEDDING_SIZE = 384

# Recursive splitter settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Number of multiprocessing workers
NUM_WORKERS = max(
    min(multiprocessing.cpu_count() // 2, 2),
    1
)


# ============================================================
# GET LOADER
# ============================================================

def get_loader(file_path):
    """
    Loads a file.

    For PDFs:
        - Extract all pages
        - Add [PAGE X] markers
        - Merge pages into one continuous document

    This allows the RecursiveCharacterTextSplitter
    to create chunks that cross PDF page boundaries.
    """

    ext = os.path.splitext(file_path)[1].lower()

    try:

        # ====================================================
        # PDF
        # ====================================================

        if ext == ".pdf":

            pages = PyPDFLoader(file_path).load()

            if not pages:
                return None

            complete_text = []

            for page_number, page in enumerate(
                pages,
                start=1
            ):

                page_text = page.page_content.strip()

                if not page_text:
                    continue

                # Add page marker
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

            # Copy metadata from first page
            if pages[0].metadata:
                metadata.update(
                    pages[0].metadata
                )

            metadata["source"] = file_path
            metadata["total_pages"] = len(pages)
            metadata["file_type"] = "pdf"

            return [{
                "page_content": full_text,
                "metadata": metadata
            }]

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

                results.append({
                    "page_content":
                        document.page_content,

                    "metadata": {
                        **document.metadata,
                        "source": file_path,
                        "file_type": "txt"
                    }
                })

            return results

        # ====================================================
        # DOC / DOCX
        # ====================================================

        elif ext in [".docx", ".doc"]:

            documents = (
                UnstructuredWordDocumentLoader(
                    file_path
                ).load()
            )

            results = []

            for document in documents:

                results.append({
                    "page_content":
                        document.page_content,

                    "metadata": {
                        **document.metadata,
                        "source": file_path,
                        "file_type": ext[1:]
                    }
                })

            return results

        # ====================================================
        # Unsupported file
        # ====================================================

        return None

    except Exception as e:

        print(
            f"\nError loading file: "
            f"{file_path}\n"
            f"{e}"
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

    for file_name in os.listdir(folder_path):

        file_path = os.path.join(
            folder_path,
            file_name
        )

        if not os.path.isfile(file_path):
            continue

        ext = os.path.splitext(
            file_name
        )[1].lower()

        if ext in supported_extensions:

            files.append(file_path)

    return files


# ============================================================
# LOAD BART MODEL
# ============================================================

def load_big_model(worker_state):

    tokenizer = AutoTokenizer.from_pretrained(
        BART_MODEL
    )

    summarizer = pipeline(
        "summarization",

        model=BART_MODEL,

        tokenizer=tokenizer,

        device=-1
    )

    worker_state["tokenizer"] = tokenizer

    worker_state["summarizer"] = summarizer


# ============================================================
# CREATE RECURSIVE TEXT SPLITTER
# ============================================================

def create_text_splitter():

    splitter = RecursiveCharacterTextSplitter(

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

    return splitter


# ============================================================
# FIND PAGE NUMBERS
# ============================================================

def get_page_numbers(text):

    """
    Extract page numbers from [PAGE X] markers.

    Example:

        [PAGE 1]
        Some text...

        [PAGE 2]
        More text...

    If a chunk contains text from both pages,
    this function returns:

        [1, 2]
    """

    import re

    matches = re.findall(
        r"\[PAGE\s+(\d+)\]",
        text
    )

    page_numbers = []

    for match in matches:

        page_number = int(match)

        if page_number not in page_numbers:

            page_numbers.append(
                page_number
            )

    return page_numbers


# ============================================================
# CLEAN PAGE MARKERS
# ============================================================

def clean_page_markers(text):

    """
    Removes [PAGE X] markers from the actual text
    before sending it to BART.
    """

    import re

    text = re.sub(
        r"\[PAGE\s+\d+\]",
        "",
        text
    )

    return text.strip()


# ============================================================
# SPLIT DOCUMENTS
# ============================================================

def split_documents(documents):

    """
    Converts complete documents into cross-page
    semantic chunks.
    """

    splitter = create_text_splitter()

    split_docs = []

    for document in documents:

        full_text = document[
            "page_content"
        ]

        metadata = document[
            "metadata"
        ]

        if not full_text.strip():
            continue

        # ----------------------------------------------------
        # Recursive splitting
        # ----------------------------------------------------

        chunks = splitter.split_text(
            full_text
        )

        # ----------------------------------------------------
        # Create chunk objects
        # ----------------------------------------------------

        for chunk_index, chunk in enumerate(
            chunks
        ):

            if not chunk.strip():
                continue

            # Determine pages represented
            page_numbers = get_page_numbers(
                chunk
            )

            # Remove page markers before
            # storing actual text
            clean_text = clean_page_markers(
                chunk
            )

            if not clean_text:
                continue

            chunk_metadata = {
                **metadata,

                "chunk_index":
                    chunk_index,

                "page_numbers":
                    page_numbers,

                "start_page":
                    min(page_numbers)
                    if page_numbers
                    else None,

                "end_page":
                    max(page_numbers)
                    if page_numbers
                    else None,
            }

            split_docs.append({

                "page_content":
                    clean_text,

                "metadata":
                    chunk_metadata
            })

    return split_docs


# ============================================================
# SUMMARIZE CHUNK
# ============================================================

def summarize(
    worker_state,
    page_content,
    metadata
):

    summarizer = worker_state[
        "summarizer"
    ]

    # text = document[
        # "page_content"
    # ].strip()
    text = page_content.strip()

    # metadata = document[
        # "metadata"
    # ]

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

        # ----------------------------------------------------
        # Very small chunks
        # ----------------------------------------------------

        if input_length < 30:

            return {
                "text": text,

                "original_text": text,

                "metadata": metadata
            }

        # ----------------------------------------------------
        # Summary length
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # BART summarization
        # ----------------------------------------------------

        result = summarizer(

            text,

            max_length=max_len,

            min_length=min_len,

            do_sample=False,

            truncation=True
        )

        summary_text = result[
            0
        ][
            "summary_text"
        ]

        return {

            # Summary
            "text":
                summary_text,

            # ORIGINAL chunk
            "original_text":
                text,

            # Metadata
            "metadata":
                metadata
        }

    except Exception as e:

        print(
            f"\nSummarization error: "
            f"{e}"
        )

        # Fallback
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

def modify_docs(documents):

    results = []

    metadata_set = set()

    for document in documents:

        metadata = document[
            "metadata"
        ]

        result = {

            # Summary
            "text":
                document["text"],

            # Original chunk
            "original_text":
                document["original_text"]
        }

        for key, value in metadata.items():

            # ------------------------------------------------
            # Source
            # ------------------------------------------------

            if key == "source":

                file_path = value

                ext = os.path.splitext(
                    file_path
                )[1].lower()[1:]

                result["source"] = ext

                result["file_path"] = (
                    file_path
                )

                metadata_set.add(
                    ("source", ext)
                )

            # ------------------------------------------------
            # Everything else
            # ------------------------------------------------

            else:

                result[key] = value

        results.append(
            result
        )

    return results, metadata_set


# ============================================================
# LOAD + SPLIT + SUMMARIZE
# ============================================================

def get_list_of_dicts(
    folder_path,
    progress_bar
):

    # ========================================================
    # 1. GET FILES
    # ========================================================

    files = get_files(
        folder_path
    )

    print(
        f"\nFound {len(files)} files."
    )

    # ========================================================
    # 2. LOAD FILES
    # ========================================================

    with WorkerPool(
        n_jobs=NUM_WORKERS,
        daemon=False
    ) as pool:

        loaded_results = pool.map(

            get_loader,

            files,

            progress_bar=progress_bar
        )

    # ========================================================
    # 3. FLATTEN DOCUMENTS
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
    # 4. CROSS-PAGE RECURSIVE SPLITTING
    # ========================================================

    split_docs = split_documents(
        documents
    )

    print(
        f"Created {len(split_docs)} "
        f"cross-page chunks."
    )

    # ========================================================
    # 5. SUMMARIZATION
    # ========================================================

    with WorkerPool(
        n_jobs=NUM_WORKERS,

        daemon=False,

        use_worker_state=True

    ) as pool:

        summarized_docs = pool.map(

            summarize,

            split_docs,

            progress_bar=progress_bar,

            worker_init=load_big_model
        )

    print(
        f"Summarized "
        f"{len(summarized_docs)} chunks."
    )

    # ========================================================
    # 6. MODIFY METADATA
    # ========================================================

    results, metadata_set = modify_docs(
        summarized_docs
    )

    return results, metadata_set


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
    # LOAD + SPLIT + SUMMARIZE
    # ========================================================

    documents, metadata_set = (
        get_list_of_dicts(
            folder_path,
            progress_bar
        )
    )

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
    # EMBED ORIGINAL CHUNKS
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

    embeddings = embedding_model.encode(

        original_texts,

        show_progress_bar=True
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
                embeddings[index].tolist(),

            payload={

                # Summary
                "text":
                    document["text"],

                # Original text
                "original_text":
                    document[
                        "original_text"
                    ],

                # File
                "file_path":
                    document.get(
                        "file_path"
                    ),

                # File type
                "source":
                    document.get(
                        "source"
                    ),

                # Chunk information
                "chunk_index":
                    document.get(
                        "chunk_index"
                    ),

                # Page traceability
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

                # Other metadata
                **{
                    key: value
                    for key, value
                    in document.items()
                    if key not in [
                        "text",
                        "original_text",
                        "file_path",
                        "source",
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

        points=points
    )

    print(
        f"Inserted {len(points)} "
        f"points into Qdrant."
    )

    # ========================================================
    # SEARCH FUNCTION
    # ========================================================

    def search_with_filter(
        query,
        query_filter,
        top_k
    ):

        query_vector = (
            embedding_model.encode(
                query
            ).tolist()
        )

        results = client.query_points(

            collection_name=
                COLLECTION_NAME,

            query=
                query_vector,

            limit=
                top_k,

            with_payload=
                True,

            query_filter=
                query_filter

        ).points

        return results

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

    # ========================================================
    # If there are filters
    # ========================================================

    if must_list:

        filters = Filter(
            should=must_list
        )

    else:

        filters = None

    # ========================================================
    # VECTOR SEARCH
    # ========================================================

    print(
        "\nSearching Qdrant..."
    )

    results = search_with_filter(

        query,

        filters,

        top_k
    )

    # ========================================================
    # RERANK
    # ========================================================

    if top_k > 100:

        return results

    if not results:

        return []

    print(
        "\nReranking results..."
    )

    reranker = CrossEncoder(
        RERANKER_MODEL
    )

    rerank_candidates = []

    for result in results:

        document_text = (
            result.payload[
                "original_text"
            ]
        )

        rerank_candidates.append(

            (
                query,
                document_text
            )
        )

    # ========================================================
    # RERANK SCORES
    # ========================================================

    rerank_scores = reranker.predict(

        rerank_candidates
    )

    reranked_results = []

    for result, score in zip(
        results,
        rerank_scores
    ):

        reranked_results.append({

            "qdrant_score":
                result.score,

            "rerank_score":
                float(score),

            "payload":
                result.payload
        })

    # ========================================================
    # SORT
    # ========================================================

    reranked_results.sort(

        key=lambda x:
            x["rerank_score"],

        reverse=True
    )

    # ========================================================
    # TOP K
    # ========================================================

    return reranked_results[
        :top_k
    ]


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    folder_path = (
        r"C:\Users\gprak\Downloads"
        # r"\Github Repos"
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

    results = create_rag_pipeline(

        folder_path,

        progress_bar,

        COLLECTION_NAME,

        EMBEDDING_SIZE,

        query,

        top_k
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n")
    print("=" * 100)
    print("FINAL SEARCH RESULTS")
    print("=" * 100)

    for index, result in enumerate(
        results,
        start=1
    ):

        payload = result[
            "payload"
        ]

        print("\n")
        print("-" * 100)

        print(
            f"RESULT {index}"
        )

        print("-" * 100)

        print(
            "Qdrant Score:",
            result[
                "qdrant_score"
            ]
        )

        print(
            "Rerank Score:",
            result[
                "rerank_score"
            ]
        )

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

        print(
            "\nSUMMARY:"
        )

        print(
            payload.get(
                "text"
            )
        )

        print(
            "\nORIGINAL CHUNK:"
        )

        print(
            payload.get(
                "original_text"
            )
        )
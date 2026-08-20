import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader

from pprint import pprint

import mpire
os.environ["OMP_NUM_THREADS"] = "1"
import time
import multiprocessing 
from mpire import WorkerPool
from pprint import pprint
from multiprocessing import Manager
from transformers import pipeline, AutoTokenizer
import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchAny, PayloadSchemaType
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from groq import Groq
import json

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
def get_loader(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext==".pdf":
        return PyPDFLoader(file_path).load()
    if ext==".txt":
        return TextLoader(file_path,encoding="utf-8").load()
    if ext in [".docx",".doc"]:
        return UnstructuredWordDocumentLoader(file_path).load()
    return None

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

def get_files(folder_path):
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]
    return files

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


def get_list_of_dicts(folder_path,progress_bar):
    files = get_files(folder_path)
    num_cores = min(multiprocessing.cpu_count()//2,2)

    results = []
    for tmp_result in files:
        my_dict = {
            "file_path":tmp_result
        }
        results.append(my_dict)
    with WorkerPool(n_jobs=num_cores,daemon=False) as pool:
        results = pool.map(get_loader, results, progress_bar = progress_bar)
    results2 = []
    for result in results:
        if result==None:
            continue
        results2+=result
    

    results = results2
    results2 = []
    results3 = []
    for result in results:
        my_dict = {
            "page_content":result.page_content
            
        }
        my_dict2 = {
            "metadata":result.metadata
        }
        results2.append(my_dict)
        results3.append(my_dict2)
    results = results2
    with WorkerPool(n_jobs=num_cores,daemon=False,use_worker_state=True) as pool:
        results = pool.map(summarize, results, progress_bar = progress_bar, worker_init=load_big_model)
        
    results2 = []
    it= 0
    for result in results:
        my_dict = {
            "text":result,
            "metadata":results3[it]["metadata"]
        }
        it+=1
        results2.append(my_dict)

    results = modify_docs(results2)
    
    return results

def modify_docs(documents):
    rv = []
    st = set()
    for my_dict in documents:
        my_dict2 = {
            "text":my_dict["text"]
        }
        for k,v in my_dict["metadata"].items():
            if k == "source":
                file_path = v
                ext = os.path.splitext(file_path)[1].lower()[1:]
                my_dict2[k] = ext
                my_dict2["file_path"] = v
                st.add((k,ext))
            else:
                my_dict2[k] = v
            

        rv.append(my_dict2)
    return rv,st

def create_rag_pipeline(documents,st,COLLECTION_NAME,EMBEDDING_SIZE,query,top_k):
    
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)


    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_SIZE,
            distance=Distance.COSINE,
        ),
    )

    key = "source"
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name=key,
        field_schema=PayloadSchemaType.KEYWORD,
    )
    
    model = SentenceTransformer("all-MiniLM-L6-v2") #384

    texts = [document["text"] for document in documents]


    embeddings = model.encode(texts)

    

    
    points = []

    for i in range(len(documents)):
        
        point = PointStruct(
            id=i + 1,
            vector=embeddings[i].tolist(),
            payload=documents[i]
        )

        points.append(point)


    
    client.upsert( #upload+insert
        collection_name=COLLECTION_NAME,
        points=points
    )

    
    def search_with_filter(query, query_filter, top_k):


        query_vector = model.encode(query).tolist()


        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        ).points


        return results
    must_list = []
    for (k,v) in st:
        m = FieldCondition(
            key=f"{k}",
            match=MatchValue(value=f"{v}")
        )
        must_list.append(m)

    filters = Filter(
        should=must_list
    )


    results = search_with_filter(query, filters,top_k)

    


    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    
    rerank_candidates = []

    for result in results:
        document_text = result.payload["text"]

        rerank_candidates.append(
            (query, document_text)
        )

    rerank_scores = reranker.predict(rerank_candidates)

    reranked_results = []

    for result, rerank_score in zip(results, rerank_scores):
        reranked_results.append({
            "qdrant_score": result.score,
            "rerank_score": float(rerank_score),
            "payload": result.payload
        })

    reranked_results.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    results = []
    for result in reranked_results[:top_k]:
        t = result
        results.append(t)
    

    return results

if __name__=="__main__":
    folder_path = r"C:\Users\gprak\Downloads\Github Repos"
    progress_bar = True
    results,st = get_list_of_dicts(folder_path,progress_bar)
    COLLECTION_NAME = "knowledge_filter"
    EMBEDDING_SIZE = 384
    simple_query = "Give details about python."
    top_k = 3
    results = create_rag_pipeline(results,st,COLLECTION_NAME,EMBEDDING_SIZE,simple_query,top_k)
    print(results)
# RAG-
# 0. Use recursive text splitter.(input take any pdf)(Done)
# 1. Split into chunks using summarization.()
# 2. Cluster into topics. Before that use bertopic to find number of topics.
# 3. Reroute to a particular topic.
# 4. Do Sparse + Dense Retrieval.(hybrid search)
# 5. Give confidence scores as well.(BM25)
# 6. After that do reranking mechanism.

import scalableOcrs 
from scalableOcrs import ocr
from pprint import pprint
import nltk
from nltk.tokenize import sent_tokenize
from bertopic import BERTopic
from transformers import pipeline




def summarize_topic(documents):

    # Combine representative documents
    combined_text = " ".join(documents)

    # BART has a limited input length.
    # Limit the text to approximately 3000 characters.
    combined_text = combined_text[:3000]

    # If text is too short, return it directly
    if len(combined_text.split()) < 20:
        return combined_text

    result = summarizer(
        combined_text,
        max_length=100,
        min_length=20,
        do_sample=False
    )

    return result[0]["summary_text"]

# ============================================================
# 6. FUNCTION TO SUMMARIZE A TOPIC
# ============================================================

def scale_summarize_topic(texts):


    topic_model = BERTopic(
        verbose=True
    )


    # ============================================================
    # 3. TRAIN BERTopic
    # ============================================================

    topics, probabilities = topic_model.fit_transform(texts)


    # ============================================================
    # 4. DISPLAY TOPIC INFORMATION
    # ============================================================

    topic_info = topic_model.get_topic_info()

    # print("\n" + "=" * 80)
    # print("TOPIC INFORMATION")
    # print("=" * 80)

    # print(topic_info)


    # ============================================================
    # 5. LOAD SUMMARIZATION MODEL
    # ============================================================

    # print("\nLoading summarization model...")

    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )
    # ============================================================
    # 7. SUMMARIZE EACH TOPIC
    # ============================================================

    topic_summaries = {}

    # print("\n" + "=" * 80)
    # print("TOPIC SUMMARIES")
    # print("=" * 80)


    for topic in topic_info["Topic"]:

        # Topic -1 represents outliers
        if topic == -1:
            continue

        # Get representative documents
        representative_docs = topic_model.get_representative_docs(topic)

        # Generate summary
        summary = summarize_topic(representative_docs)

        topic_summaries[topic] = summary

        # Get topic keywords
        topic_words = topic_model.get_topic(topic)

        keywords = [
            word
            for word, score in topic_words[:10]
        ]

        # print("\n" + "-" * 80)
        # print(f"TOPIC: {topic}")
        # print("-" * 80)

        # print("Keywords:")
        # print(", ".join(keywords))

        # print("\nRepresentative documents:")

        # for doc in representative_docs:
            # print("  -", doc.strip())

        # print("\nSummary:")
        # print(summary)



    # ============================================================
    # 8. SHOW WHICH TOPIC EACH ORIGINAL TEXT BELONGS TO
    # ============================================================

    # print("\n" + "=" * 80)
    # print("DOCUMENT → TOPIC")
    # print("=" * 80)

    # for i, (text, topic) in enumerate(zip(texts, topics)):

        # print(f"\nDocument {i + 1}")
        # print("Topic:", topic)
        # print("Text:", text.strip())

    return {"topics":topics}
    # BART has a limited input length.
    # Limit the text to approximately 3000 characters.
    # combined_text = combined_text[:3000]

    # # If text is too short, return it directly
    # if len(combined_text.split()) < 20:
    #     return combined_text
    # input_length = combined_text.split()
    # max_len = min(100,max(30,int(input_length*0.6)))
    # min_len = int(max_len*0.4)


    # result = summarizer(
    #     combined_text,
    #     max_length=max_len,
    #     min_length=min_len,
    #     do_sample=False,
    #     truncation = True,
    #     num_beans = 4,
    #     length_penalty = 1.8

    # )

    # return result[0]["summary_text"]




def split_text(text):
    sentences = sent_tokenize(text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= 500:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
    


def extract_text(pdf_paths,det_arch,reco_arch,progress_bar):
    texts = ocr.extract_text_from_pdfs(pdf_paths,det_arch,reco_arch,progress_bar)
    return texts

if __name__=="__main__":

    pdf_paths = [
        r"C:\Users\gprak\Downloads\PrakharGandhi_FullStackAIEngineer_TechnicalLeadArchitect_Accenture.pdf"
    ]
    det_arch="linknet_resnet50"
    reco_arch="crnn_vgg16_bn"  
    progress_bar = True  
    texts = extract_text(pdf_paths,det_arch,reco_arch,progress_bar)
    # pprint(texts)

    texts_list = [my_dict["extracted_text"] for my_dict in texts] 
    text = "\n".join(texts_list)
    chunks = split_text(text)

    result = scale_summarize_topic(chunks)
    pprint(result)
    # summarize_texts = []
    # for text in chunks:
    #     out = scale_summarize_topic(text)
    #     summarize_texts.append(out)
    # print(summarize_texts)


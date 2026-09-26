import glob
import json
from typing import List
import chromadb
import mlflow
from mlflow.genai import scorer
from mlflow.entities import Document
from mlflow.genai.scorers import (
    Correctness,
    RetrievalSufficiency,
    RetrievalRelevance,
    RetrievalGroundedness,
)
from openai import OpenAI
import numpy as np




# Clients


client_op = OpenAI()

client_chroma = chromadb.HttpClient(
    host="localhost",
    port=8000,
)



# Chroma collections


fixed_chunk_collection = client_chroma.get_or_create_collection(
    "fixed_chunk_collection"
)

chunk_by_topic_collection = client_chroma.get_or_create_collection(
    "chunk_by_topic_collection"
)

chunk_topic_item_collection = client_chroma.get_or_create_collection(
    "chunk_by_topic_item_collection"
)


# MLflow setup


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Evaluating Chunk Strategy")



# Retriever


@mlflow.trace(span_type="RETRIEVER")
def retrieve(
    collection,
    question: str,
    n_results: int = 3,
) -> List[Document]:

    results = collection.query(
        query_texts=[question],
        n_results=n_results,
    )

    return [
        Document(
            page_content=doc,
            metadata={
                "doc_id": doc_id,
            },
        )
        for doc, doc_id in zip(
            results["documents"][0],
            results["ids"][0],
        )
    ]

# cosine similarity func
def cosine_sim(a,b):
    a = np.array(a)
    b = np.array(b)
    
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )
    
def get_embedding(text:str):
    response = client_op.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )

    return response.data[0].embedding

@scorer
def semantic_similarity(trace,expectations)->float:
    
    similarities = []
    
    expected_answer = expectations["expected_facts"]
    
    retrieve_spans = trace.search_spans(span_type ="RETRIEVER")
    
    if not retrieve_spans:
        return 0.0
    
    docs = retrieve_spans[-1].outputs
    
    if not docs:
        return 0.0
    

    expected_embedding = get_embedding(expected_answer)
    
    for doc in docs:
        doc_embedding = get_embedding(doc.page_content)
    
        similarity = cosine_sim(doc_embedding,expected_embedding)
        
        similarities.append(similarities)
    
    #return best retrieved chunk 
    return max(similarities)



#k=3
@scorer
def hit_at_k(trace,expectations: dict)-> bool:
    
    expected_doc_id = str(expectations["expected_doc_id"])
    
    #get all the retriever spans in the given trace
    retrieve_spans = trace.search_spans(span_type ="RETRIEVER")
    
    if not retrieve_spans:
        return False
    
    #get the documents which were the output othe retriever span in this case
    docs = retrieve_spans[-1].outputs
    
    if docs:
        for doc in docs:
            if str(doc.metadat["doc_id"]) == expected_doc_id:
                return True
    
    return False
    
    


# Shared RAG function


def answer_question(collection, question: str):

    docs = retrieve(
        collection,
        question,
    )

    if not docs:
        return {
            "response": "I don't have information to answer that question."
        }

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Answer the user's question based only on "
                f"the following context:\n\n{context}"
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    response = client_op.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )

    return {
        "response": response.choices[0].message.content
    }



# Three chunking strategies


@mlflow.trace(span_type="AGENT")
def rag_agent_fixed(question: str):
    return answer_question(
        fixed_chunk_collection,
        question,
    )


@mlflow.trace(span_type="AGENT")
def rag_agent_topic(question: str):
    return answer_question(
        chunk_by_topic_collection,
        question,
    )


@mlflow.trace(span_type="AGENT")
def rag_agent_item(question: str):
    return answer_question(
        chunk_topic_item_collection,
        question,
    )


if __name__ == " __main__":
    
    # Evaluation dataset


    paths = (
        "/Users/dillyejeh/Documents/Career/August 2026 cycle/"
        "Meeting Notes Extractor/data/QMSum/data/ALL/train/*json"
    )

    eval_data = []

    for doc_id, meeting_json in enumerate(glob.glob(paths)):

        with open(meeting_json, "r") as f:
            meeting_dict = json.load(f)

        for item in meeting_dict["specific_query_list"]:

            eval_data.append(
                {
                    "inputs": {
                        "question": item["query"]
                    },
                    "expectations": {
                        "expected_facts": [item["answer"]],
                        "expected_doc_id": doc_id
                    },
                    
                }
            )


    # Scorers


    scorers = [
        Correctness(),
        RetrievalSufficiency(),
        RetrievalRelevance(),
        RetrievalGroundedness(),
        hit_at_k, 
        semantic_similarity
    ]


    # Evaluate each strategy

    fixed_results = mlflow.genai.evaluate(
        data=eval_data,
        predict_fn=rag_agent_fixed,
        scorers=scorers,
    )

    topic_results = mlflow.genai.evaluate(
        data=eval_data,
        predict_fn=rag_agent_topic,
        scorers=scorers,
    )

    item_results = mlflow.genai.evaluate(
        data=eval_data,
        predict_fn=rag_agent_item,
        scorers=scorers,
    )



    
    

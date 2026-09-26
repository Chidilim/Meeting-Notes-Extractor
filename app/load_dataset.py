import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import glob
import json
from extraction.ingestion import get_meeting_data
from schemas.extraction_schema import MeeetingNote, ChunkStrat, ChunkStratCater
import os

client = chromadb.HttpClient(host="localhost", port=8000)


if __name__ == "__main__":
    
        #CREATING THE DIFFERENT COLLECTIONS FOR EACH CHUNKING STRATEGY
    fixed_chunk_collection = client.get_or_create_collection(
        name="fixed_chunk_collection",
        embedding_function=OpenAIEmbeddingFunction(
            model_name="text-embedding-3-small"
        )
    )

    chunk_by_topic_collection = client.get_or_create_collection(
        name="chunk_by_topic_collection",
        embedding_function=OpenAIEmbeddingFunction(
            model_name="text-embedding-3-small"
        )
    )


    chunk_topic_item__collection = client.get_or_create_collection(
        name="chunk_by_topic_item_collection",
        embedding_function=OpenAIEmbeddingFunction(
            model_name="text-embedding-3-small"
        )
    )

    strategy1 = ChunkStrat(chunk_strategy=ChunkStratCater.FIXED)
    strategy2 = ChunkStrat(chunk_strategy=ChunkStratCater.TOPIC)
    strategy3 = ChunkStrat(chunk_strategy=ChunkStratCater.ITEM)
    
    paths = glob.glob("/Users/dillyejeh/Documents/Career/August 2026 cycle/Meeting Notes Extractor/data/QMSum/data/ALL/train/*json")
    
    for doc_id, path in enumerate(paths):
        with open(path, "r") as f:
            meeting_dict = json.load(f)
            data = MeeetingNote(meeting=meeting_dict)
            
            fixed_chunks = get_meeting_data(data,strategy1)
            topic_chunks = get_meeting_data(data,strategy2)
            item_chunks = get_meeting_data(data,strategy3)
            
            
            fixed_chunk_collection.upsert(ids=[f"document_{doc_id}_chunk_{i}" for i in range(len(fixed_chunks))], documents=fixed_chunks,metadatas=[
            {
                "document_id": doc_id,
                "chunk_index": i
            }
            for i in range(len(fixed_chunks))
        ])
            
            
            chunk_by_topic_collection.upsert(ids=[f"document_{doc_id}_chunk_{i}" for i in range(len(topic_chunks))], documents= topic_chunks, metadatas=[
            {
                "document_id": doc_id,
                "chunk_index": i
            }
            for i in range(len(topic_chunks))
        ])
            
            
            chunk_topic_item__collection.upsert(
        ids=[
            f"document_{doc_id}_item_{i}"
            for i in range(len(item_chunks))
        ],

        documents=[
            f"{item.type}: {item.text}. Owner: {item.owner or 'unknown'}"
            for item in item_chunks
        ],

        metadatas=[
            {
                "document_id": doc_id,
                "item_index": i,
                "type": item.type,
                "owner": item.owner or "unknown"
            }
            for i, item in enumerate(item_chunks)
        ]
    )       
            
    
        


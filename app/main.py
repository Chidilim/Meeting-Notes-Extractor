#in here we would have our ingest and query pipeline using FAST API
#FAST API AND CHROMADB talk to themselves using docker compose
#chromadb - using port 800
#here connect the routers ingestion and query into one 

from fastapi import FastAPI

app = FastAPI()
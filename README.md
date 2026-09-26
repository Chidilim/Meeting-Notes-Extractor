# Meeting Notes Extractor

A Python-based meeting intelligence prototype for turning meeting transcripts into searchable, structured information. The project experiments with different transcript chunking strategies and evaluates their usefulness for retrieval-augmented question answering (RAG).

> **Project status:** This repository is under active development. The FastAPI application and ingestion/query endpoints are currently scaffolding, while the transcript ingestion, chunking, structured extraction, and retrieval experiments are being built out.

## What it does

The project currently supports experimentation with three ways to represent meeting transcripts:

- **Fixed-size chunks** – splits the transcript into token-sized sections.
- **Topic chunks** – groups transcript turns using the meeting's topic spans.
- **Extracted items** – uses an OpenAI model to extract action items and decisions, including an optional owner.

The extracted data can be stored in ChromaDB collections for semantic retrieval. The experiment code uses MLflow GenAI evaluation tools to compare retrieval and answer quality across chunking strategies.

## Project structure

```text
.
├── app/
│   ├── main.py          # FastAPI application entry point
│   ├── load_dataset.py  # Loads meeting data into ChromaDB
│   ├── ingest.py        # Ingestion pipeline placeholder
│   └── query.py         # Query pipeline placeholder
├── extraction/
│   └── ingestion.py     # Transcript formatting, chunking, and item extraction
├── schemas/
│   └── extraction_schema.py  # Pydantic models and chunking strategy definitions
├── experiments/
│   └── experiment_chunking.py # RAG and chunking-strategy evaluation
├── tests/
│   └── test_chunking.py # Chunking and transcript construction tests
├── compose.yaml         # API and ChromaDB services
└── dockerfile           # Container configuration
```

## Requirements

- Python 3.12 or later
- An OpenAI API key for embeddings and structured item extraction
- Docker and Docker Compose, if running ChromaDB through the provided Compose configuration
- Meeting data in the expected QMSum-style JSON format

The code currently uses libraries including FastAPI, ChromaDB, OpenAI, Pydantic, `tiktoken`, `python-dotenv`, MLflow, NumPy, and pytest.

## Configuration

Create a `.env` file in the repository root and add your OpenAI key:

```env
OPENAI_API_KEY=your-openai-api-key
```

Do not commit `.env` or API keys to version control.

## Running ChromaDB

Start the ChromaDB service with Docker Compose:

```bash
docker compose up chromadb
```

ChromaDB is expected to be available at `http://localhost:8000`.

## Running the API

Install the project dependencies in a virtual environment, then start the FastAPI application:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install fastapi uvicorn chromadb openai pydantic tiktoken python-dotenv
uvicorn app.main:app --reload --port 8080
```

The API will be available at:

- http://localhost:8080
- http://localhost:8080/docs – Swagger UI

The API routes are still being implemented; the current application provides the FastAPI foundation.

## Loading meeting data

`app/load_dataset.py` creates ChromaDB collections for the fixed-size, topic-based, and item-based strategies. To use it:

1. Place QMSum-style meeting JSON files under the data directory expected by the script.
2. Start ChromaDB.
3. Set `OPENAI_API_KEY` in `.env`.
4. Update any local dataset paths in `app/load_dataset.py`.
5. Run:

```bash
python app/load_dataset.py
```

Each meeting is transformed into chunks and upserted with document and chunk metadata.

## Running experiments

The experimental RAG pipeline is in `experiments/experiment_chunking.py`. It retrieves context from ChromaDB, asks an OpenAI model to answer questions using that context, and evaluates strategies with MLflow.

The experiment expects:

- ChromaDB at `localhost:8000`
- MLflow at `localhost:5000`
- An OpenAI API key
- QMSum-style evaluation data

Update the dataset paths and service configuration for your environment before running the experiment.

## Running tests

Install pytest and run:

```bash
pytest
```

Some tests expect the QMSum dataset to exist at the configured `data/QMSum/...` path.

## Data format

The ingestion pipeline expects a meeting object containing transcript turns and topics similar to:

```json
{
  "meeting_transcripts": [
    {"speaker": "Speaker 1", "content": "Meeting discussion..."}
  ],
  "topic_list": [
    {
      "topic": "Project planning",
      "relevant_text_span": [[0, 4]]
    }
  ]
}
```

The exact topic field names should match the source dataset used by the ingestion code.

## Roadmap

- Complete FastAPI ingestion and query routes.
- Remove environment-specific filesystem paths from scripts.
- Add a reproducible dependency file and deployment configuration.
- Add end-to-end tests with a small fixture dataset.
- Compare chunking strategies using consistent evaluation data and metrics.
- Improve extraction reliability and error handling.

## Contributing

Contributions and experiments are welcome. Please open an issue to discuss larger changes before submitting a pull request. Keep credentials, local dataset files, generated artifacts, and machine-specific paths out of commits.

## License

No license has been specified yet. Add a license before distributing this project or accepting external contributions under defined terms.

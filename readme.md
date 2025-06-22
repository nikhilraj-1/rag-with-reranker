# RAG App - Local Retrieval-Augmented Generation with HuggingFace, Qdrant, and LangChain
## A fast, fully local Retrieval-Augmented Generation (RAG) system built using open-source components:

### Embedding Model: BAAI/bge-large-en-v1.5
### LLM: TinyLlama/TinyLlama-1.1B-Chat-v1.0
### Vector Store: Embedded Qdrant (no cloud API needed)
### Reranking: `cross-encoder/ms-marco-MiniLM-L-6-v2`
### Interface: FastAPI (`main.py`)
### Framework: LangChain

## Project Structure

<pre> ``` rag_app/ ├── app/ │ ├── config.py # Loads and validates config.yaml │ ├── ingest.py # Parses, splits, embeds, and stores PDFs │ ├── models.py # Loads embedding + LLM models │ ├── query.py # RAG pipeline with deduplication + reranking │ ├── logger.py # Custom logger with formatting │ └── evaluate.py # Automated evaluation with BLEU, ROUGE, BERTScore ├── config/ │ └── config.yaml # All configs (models, paths, etc.) ├── data/ │ └── documents/ # Place your PDF documents here ├── main.py # FastAPI endpoint ├── test/ │ ├── test_dataset.csv # Test Q&A pairs for eval │ └── qag_score_results.json # Output of evaluation ├── requirements.txt └── README.md ``` </pre>


## Setup
1. Clone + Install Dependencies
clone from :  https://github.com/nikhilraj-1/rag-with-reranker
cd rag_app
pip install -r requirements.txt

2. Prepare Config
Edit config/config.yaml:

3. Place PDFs
Put your documents inside ./data/documents/.

## Run
1. Ingest and Index PDFs

python -m app.ingest

This will Load and split PDFs, Embed chunks and Store them in local Qdrant.

2. Start FastAPI Server

uvicorn main:app --reload
Go to http://localhost:8000.

## Query it via:

GET /ask?q=your+question

Example:

curl "http://localhost:8000/ask?q=What+is+hypertension?"

# How It Works

                          ┌──────────────────────────┐
                          │        config.yaml       │
                          │  (Model names, paths)    │
                          └────────────┬─────────────┘
                                       │
          ┌────────────────────────────▼───────────────────────────┐
          │                    PDF Ingestion Flow                  │
          └────────────────────────────────────────────────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │   PDF_DIR (./data)      │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │   PyMuPDFLoader         │
                          │   (Loads PDFs)          │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │ RecursiveCharacterSplitter│
                          │ (Chunk: size=512, overlap=50)│
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │ HuggingFaceEmbeddings   │
                          │ (BAAI/bge-large-en-v1.5)│
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │     Qdrant (local)      │
                          │ Vectors stored by chunks│
                          └─────────────────────────┘


          ┌────────────────────────────▲────────────────────────────┐
          │                   Runtime Query Flow                    │
          └─────────────────────────────────────────────────────────┘
                                       │
                            ┌──────────┴──────────┐
                            │  query_engine(query)│
                            └──────────┬──────────┘
                                       │
                            ┌──────────▼──────────┐
                            │ Embed query → vector│
                            └──────────┬──────────┘
                                       │
                       ┌───────────────▼───────────────┐
                       │   Qdrant.similarity_search()  │
                       │     (top-k = 10 docs)         │
                       └───────────────┬───────────────┘
                                       │
                         ┌────────────▼────────────┐
                         │ Deduplicate (SequenceMatcher > 0.9) │
                         └────────────┬────────────┘
                                       │
                     ┌────────────────▼────────────────┐
                     │ CrossEncoder Reranker           │
                     │ (ms-marco-MiniLM-L-6-v2)        │
                     └────────────┬────────────────────┘
                                  │ top-2 documents
                                  ▼
                         ┌────────────────────┐
                         │ Truncate to context│
                         │ (max 3000 tokens)  │
                         └────────────┬───────┘
                                      │
                      ┌───────────────▼────────────────┐
                      │ HuggingFacePipeline LLM        │
                      │ TinyLlama-1.1B-Chat-v1.0        │
                      └───────────────┬────────────────┘
                                      │
                           ┌──────────▼──────────┐
                           │ RetrievalQA Chain   │
                           └──────────┬──────────┘
                                      │
                           ┌──────────▼──────────┐
                           │ Final Answer Output │
                           └─────────────────────┘



## Ingestion Pipeline (app/ingest.py)
PDFs parsed and chunked using CHUNK_SIZE + CHUNK_OVERLAP

Embeddings computed using BAAI/bge-large-en-v1.5

Embedded chunks stored in local Qdrant collection

## Query Flow (app/query.py)
Semantic Search via Qdrant (TOP_K)

Deduplication using cosine similarity threshold (DEDUPLICATION_THRESHOLD)

Reranking using cross-encoder (RERANK_MODEL, RERANK_TOP_K)

Filtering: Only keep chunks above SCORE_THRESHOLD

LLM Response: RAG prompt sent to TinyLlama using LangChain + HF pipeline


## Evaluation 
Place your test dataset as CSV at test/test_dataset.csv with columns:
question, expected_answer.

python evaluate.py

Metrics:

BLEU

ROUGE-L

BERTScore

Output saved to: test/qag_score_results.json

sample run on two questions : "average_bleu_score": 30.660274527412263,
  "average_rouge_l_score": 0.5617386489479512,
  "average_bert_f1_score": 0.9145451486110687,
  "average_qag_score": 68.71147180447443,
  "average_confidence": 6.922852072051732

## The metrics suggests: 

Retrieval is top-notch — high BERT-F1 and QAG back that up.
The LLM is semantically right, though surface overlap (BLEU, ROUGE) isn't as high — expected for small models.
System is reliable and accurate, even with lightweight generation.

## Features
Works entirely offline (embedded Qdrant + local Hugging Face models)

Supports CPU, MPS (Apple), and CUDA acceleration

Handles chunk deduplication and reranking (using CrossEncoder)

Plug-and-play config system

Modular codebase with separation of ingestion, querying, and evaluation

## Requirements
Python 3.9+

Memory: ~16 GB RAM recommended (esp. for reranking)

Storage: Disk space for models and vector DB

OS: macOS, Linux, or WSL2 on Windows

## Notes
Model quantization is enabled automatically if CUDA is available.

Documents with invalid or zero embeddings are skipped.

Chunk sizes and overlap can be tuned via config.yaml.

## TODO
 Streamed responses with token-by-token generation

 Add Gradio/Streamlit frontend

 UI to upload PDFs and ask questions

 Docker support for deployment

## License
MIT License. Feel free to use, modify, and share.

## Author
Nikhil Raj
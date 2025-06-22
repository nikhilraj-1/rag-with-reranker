# rag_app/app/ingest.py
from app.config import PDF_DIR, QDRANT_PATH, QDRANT_COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP
from app.models import load_embedding_model
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant as QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import os
import numpy as np
from app.logger import get_logger

logger = get_logger(__name__)


def is_valid(vec: np.ndarray) -> bool:
    return np.all(np.isfinite(vec)) and not np.allclose(vec, 0)

def ingest_documents():
    # Initialize Qdrant client with explicit closure
    client = QdrantClient(path=str(QDRANT_PATH))
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]
        if QDRANT_COLLECTION_NAME in collection_names:
            logger.info(f"[ingest] Collection '{QDRANT_COLLECTION_NAME}' already exists at '{QDRANT_PATH}'. Skipping ingestion.")
            return
    except Exception as e:
        logger.info(f"[ingest] ERROR checking collection existence: {e}. Proceeding with ingestion.")
    finally:
        client.close()  # Ensure client is closed

    if not PDF_DIR.exists():
        logger.info(f"[ingest] ERROR: PDF_DIR '{PDF_DIR}' does not exist.")
        return

    docs = []
    for fname in os.listdir(PDF_DIR):
        if fname.lower().endswith(".pdf"):
            path = PDF_DIR / fname
            try:
                loader = PyMuPDFLoader(str(path))
                docs.extend(loader.load())
            except Exception as e:
                logger.info(f"[ingest] ERROR loading PDF {fname}: {e}")
    logger.info(f"[ingest] Loaded {len(docs)} documents from {PDF_DIR}")

    if not docs:
        logger.info("[ingest] No PDF documents found; skipping ingestion.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    logger.info(f"[ingest] Split into {len(chunks)} chunks")

    embeddings = load_embedding_model()
    test_vec = embeddings.embed_query("test sentence")
    logger.info(f"[ingest] Sample Embedding valid? {is_valid(np.array(test_vec))}, Length: {len(test_vec)}")


    valid_chunks = []
    for chunk in chunks:
        try:
            chunk_embedding = embeddings.embed_documents([chunk.page_content])[0]
            if is_valid(np.array(chunk_embedding)):
                valid_chunks.append(chunk)
            else:
                logger.info(f"[ingest] WARNING: Invalid embedding for chunk: {chunk.page_content[:50]}...")
        except Exception as e:
            logger.info(f"[ingest] ERROR embedding chunk: {e}")

    logger.info(f"[ingest] {len(valid_chunks)} valid chunks out of {len(chunks)}")

    if not valid_chunks:
        logger.info("[ingest] No valid chunks to ingest; aborting.")
        return

    client = QdrantClient(path=str(QDRANT_PATH))
    try:
        # Explicitly create collection
        client.recreate_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=len(test_vec), distance=Distance.COSINE),
        )

        # Now connect via LangChain's wrapper
        qdrant_vs = QdrantVectorStore(
            client=client,
            collection_name=QDRANT_COLLECTION_NAME,
            embeddings=embeddings,
        )
        qdrant_vs.add_documents(valid_chunks)

        logger.info(f"[ingest] Ingestion complete into local Qdrant at '{QDRANT_PATH}' collection '{QDRANT_COLLECTION_NAME}'")
    except Exception as e:
        logger.info(f"[ingest] ERROR during ingestion: {e}")
    finally:
        client.close()


if __name__ == "__main__":
    ingest_documents()

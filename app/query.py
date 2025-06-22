# rag_app/app/query.py

from app.config import QDRANT_PATH, QDRANT_COLLECTION_NAME
from app.models import load_llm, load_embedding_model
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_qdrant import Qdrant as QdrantVectorStore
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
import numpy as np
import traceback
from langchain_core.documents import Document
from difflib import SequenceMatcher
from app.logger import get_logger

logger = get_logger(__name__)

def query_engine(query: str) -> str:
    embeddings = load_embedding_model()
    client = QdrantClient(path=str(QDRANT_PATH))
    try:
        vectordb = QdrantVectorStore(
            client=client,
            collection_name=QDRANT_COLLECTION_NAME,
            embeddings=embeddings
        )

        # Validate query embedding
        try:
            query_embedding = embeddings.embed_query(query)
            if not np.all(np.isfinite(query_embedding)):
                logger.info(f"[query] ERROR: Invalid query embedding for '{query}'")
                return "Error: Invalid query embedding."
        except Exception as e:
            logger.info(f"[query] ERROR generating query embedding: {e}")
            return "Error generating query embedding."

        # Initial retrieval with scores
        try:
            doc_with_scores = vectordb.similarity_search_with_score(query, k=10)
            docs = [doc for doc, _ in doc_with_scores]
            scores = [score for _, score in doc_with_scores]
            logger.info(f"[query] Retrieved {len(docs)} documents for query: '{query}'")
            logger.info(f"[query] Top-{len(docs)} Retrieved Documents (Pre-Reranking):")
            for i, (doc, score) in enumerate(zip(docs, scores), 1):
                logger.info(f"Document {i}:")
                logger.info(f"  Content: {doc.page_content}")
                logger.info(f"  Metadata: {doc.metadata}")
                logger.info(f"  Similarity Score: {score}")
                logger.info("-" * 50)
        except Exception as e:
            logger.info(f"[query] ERROR retrieving documents: {e}")
            return "Error retrieving documents."

        if not docs:
            return "No relevant documents found."

        # Remove duplicate or near-duplicate documents
        deduped_docs = []
        for doc in docs:
            content = doc.page_content.strip()
            is_dup = False
            for existing in deduped_docs:
                # use ratio threshold for near-duplicates
                if SequenceMatcher(None, content, existing.page_content.strip()).ratio() > 0.9:
                    is_dup = True
                    break
            if not is_dup:
                deduped_docs.append(doc)
        docs = deduped_docs
        logger.info(f"[query] {len(docs)} documents remain after deduplication.")
        if not docs:
            return "No relevant documents after deduplication."

        # Apply reranker
        try:
            reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            pairs = [[query, doc.page_content] for doc in docs]
            rerank_scores = reranker.predict(pairs)
            doc_score_pairs = list(zip(docs, rerank_scores))
            doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
            top_n = 2
            reranked_docs = [
                Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "score": float(score)}
                ) for doc, score in doc_score_pairs[:top_n]
            ]
            logger.info(f"[query] Top-{len(reranked_docs)} Reranked Documents:")
            for i, doc in enumerate(reranked_docs, 1):
                logger.info(f"Reranked Document {i}:")
                logger.info(f"  Content: {doc.page_content}")
                logger.info(f"  Metadata: {doc.metadata}")
                logger.info(f"  Reranker Score: {doc.metadata.get('score', 'N/A')}")
                logger.info("-" * 50)
        except Exception as e:
            logger.info(f"[query] ERROR during reranking: {e}")
            traceback.print_exc()
            return "Error during reranking."

        if not reranked_docs:
            return "No relevant documents after reranking."

        # Truncate document content for LLM
        max_context_length = 3000
        context = "\n".join([doc.page_content[:max_context_length//len(reranked_docs)] for doc in reranked_docs])

        llm = load_llm()
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
        "Answer the question using only the context below.\n"
        "Be brief, factual, and avoid repetition.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
        )
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectordb.as_retriever(search_kwargs={"k": 10}),
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt_template}
        )
        try:
            result = qa_chain.invoke({"query": query, "context": context})
            logger.info(f"[query] QA chain result: {result}")
            return result["result"]
        except Exception as e:
            logger.info(f"[query] ERROR during QA chain invocation: {e}")
            traceback.print_exc()
            return "Error during answer generation."
    finally:
        client.close() 

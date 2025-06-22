import pandas as pd
import requests
import json
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import numpy as np
from pathlib import Path
from app.logger import get_logger

logger = get_logger(__name__)


# Paths
BASE_DIR = Path("/Users/nikhilraj/Documents/rag_project")
TEST_DATASET_PATH = BASE_DIR / "test" / "test_dataset.csv"
OUTPUT_PATH = BASE_DIR / "test" / "qag_score_results.json"
test_df = pd.read_csv(TEST_DATASET_PATH)
logger.info(f"[debug] CSV columns: {test_df.columns.tolist()}")


# Query chatbot
def query_chatbot(query: str) -> dict:
    try:
        response = requests.get("http://127.0.0.1:8000/ask", params={"query": query}, timeout=300)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.info(f"[qag] ERROR querying chatbot for '{query}': {e}")
        return {"answer": "Error", "retrieved_docs": [], "reranked_docs": []}

# Evaluate QAG score
results = []
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
for _, row in test_df.iterrows():
    query = row["query"]
    ground_truth = row["ground_truth_answer"]
    response = query_chatbot(query)
    generated_answer = response.get("answer", "Error")
    
    # Initialize metrics
    bleu_score = 0.0
    rouge_l_score = 0.0
    bert_f1_score = 0.0
    reasoning = ""
    confidence = 0.0

    if generated_answer != "Error":
        # BLEU
        bleu_score = corpus_bleu([generated_answer], [[ground_truth]]).score
        
        # ROUGE-L
        rouge_scores = scorer.score(ground_truth, generated_answer)
        rouge_l_score = rouge_scores["rougeL"].fmeasure
        
        # BERTScore
        P, R, F1 = bert_score([generated_answer], [ground_truth], lang="en", device="cpu")
        bert_f1_score = F1.mean().item()
        
        # Normalize scores (0-100)
        norm_bleu = bleu_score
        norm_rouge_l = rouge_l_score * 100
        norm_bert = bert_f1_score * 100
        
        # QAG Score: Weighted average
        qag_score = (0.2 * norm_bleu + 0.3 * norm_rouge_l + 0.5 * norm_bert)
        
        # Confidence: Inverse of standard deviation of normalized scores
        norm_scores = [norm_bleu, norm_rouge_l, norm_bert]
        std_dev = np.std(norm_scores) if norm_scores else 0.0
        confidence = max(0, 100 - std_dev * 5) if std_dev is not None else 100.0
        
        # Reasoning
        if qag_score < 50:
            reasoning = "Low QAG score: Likely due to incomplete answer or retrieval of irrelevant documents."
            if bert_f1_score > 0.8 and bleu_score < 20:
                reasoning += " High BERTScore suggests semantic similarity, but low BLEU indicates paraphrasing or missing key terms."
            if rouge_l_score < 0.5:
                reasoning += " Low ROUGE-L suggests structural mismatch or missing key content."
        elif qag_score < 75:
            reasoning = "Moderate QAG score: Answer is partially correct but may lack detail or exact phrasing."
            if bert_f1_score > 0.85:
                reasoning += " High BERTScore indicates good semantic match, but lower BLEU/ROUGE-L suggests minor content gaps."
        else:
            reasoning = "High QAG score: Answer closely matches ground truth in content and semantics."
        
        if std_dev > 20:
            reasoning += " Low confidence due to high metric disagreement (e.g., BERTScore high, BLEU/ROUGE low)."
        elif std_dev < 10:
            reasoning += " High confidence due to consistent metric scores."
    else:
        reasoning = "Error in generating answer, possibly due to retrieval or processing failure."
        qag_score = 0.0
        confidence = 0.0

    result = {
        "query": query,
        "ground_truth": ground_truth,
        "generated_answer": generated_answer,
        "bleu_score": bleu_score,
        "rouge_l_score": rouge_l_score,
        "bert_f1_score": bert_f1_score,
        "qag_score": qag_score,
        "confidence": confidence,
        "reasoning": reasoning,
        "retrieved_docs": response.get("retrieved_docs", []),
        "reranked_docs": response.get("reranked_docs", []),
    }
    results.append(result)

# Aggregate metrics
valid_results = [r for r in results if r["generated_answer"] != "Error"]
if valid_results:
    avg_bleu = np.mean([r["bleu_score"] for r in valid_results])
    avg_rouge_l = np.mean([r["rouge_l_score"] for r in valid_results])
    avg_bert_f1 = np.mean([r["bert_f1_score"] for r in valid_results])
    avg_qag = np.mean([r["qag_score"] for r in valid_results])
    avg_confidence = np.mean([r["confidence"] for r in valid_results])
else:
    avg_bleu = avg_rouge_l = avg_bert_f1 = avg_qag = avg_confidence = 0.0

# Save results
report = {
    "average_bleu_score": avg_bleu,
    "average_rouge_l_score": avg_rouge_l,
    "average_bert_f1_score": avg_bert_f1,
    "average_qag_score": avg_qag,
    "average_confidence": avg_confidence,
    "results": results,
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(report, f, indent=2)

# Print summary
logger.info(f"[qag] Average QAG Score: {avg_qag:.2f}")
logger.info(f"[qag] Average BLEU Score: {avg_bleu:.2f}")
logger.info(f"[qag] Average ROUGE-L F1: {avg_rouge_l:.3f}")
logger.info(f"[qag] Average BERTScore F1: {avg_bert_f1:.3f}")
logger.info(f"[qag] Average Confidence: {avg_confidence:.2f}")
logger.info(f"[qag] Detailed results saved to {OUTPUT_PATH}")
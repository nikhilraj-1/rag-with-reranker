# rag_app/app/models.py
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from app.config import EMBEDDING_MODEL_NAME, LLM_MODEL_NAME
import torch

from app.logger import get_logger
logger = get_logger(__name__)



def load_embedding_model():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device}
    )

def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    # Only use quantization for CUDA
    quantization_config = BitsAndBytesConfig(load_in_4bit=True) if torch.cuda.is_available() else None
    if torch.cuda.is_available():
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            device_map="auto",
            torch_dtype=torch_dtype,
            quantization_config=quantization_config
        )
    except Exception as e:
        logger.info(f"[models] ERROR loading model with quantization: {e}")
        logger.info("[models] Falling back to non-quantized model")
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            device_map="auto",
            torch_dtype=torch_dtype
        )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.2,
        do_sample=True,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    return HuggingFacePipeline(
        pipeline=pipe,
        model_kwargs={"temperature": 0.2, "max_length": 1024}
    )
import os
import logging
import urllib.request
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

ANNOTATED_TSV_PATH = "results/egIwc7NRt4hou5yo.txt"
DOWNLOAD_DIR = "downloads"
ABSTRACTS_DIR = "pubmed_abstracts"
CHROMA_DB_PATH = "chroma_db"
CHROMA_COLLECTION_NAME = "pubmed_abstracts"
EMBEDDING_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
RETRIEVAL_MODEL_DEVICE = os.getenv("IGKF_RETRIEVAL_MODEL_DEVICE", "cpu")
SPACY_MODEL_NAME = "en_core_sci_lg"

LOCAL_MODEL_NAME = "Qwen3-8B"
LOCAL_MODEL_IDENTIFIER = "Qwen/Qwen3-8B-GGUF"
LOCAL_TOKENIZER_IDENTIFIER = "Qwen/Qwen3-8B"
LOCAL_GGUF_FILENAME = "Qwen3-8B-Q4_K_M.gguf"
LLM_MODEL_PATH = "./models/Qwen3-8B-Q4_K_M.gguf"
LOCAL_MODEL_QUANTIZATION = "Q4_K_M"
N_GPU_LAYERS = -1
N_CTX = 32768
N_BATCH = 512
N_UBATCH = 128
LOCAL_FLASH_ATTENTION = True
LOCAL_TEMPERATURE = 0.3
LOCAL_TOP_P = 0.8
LOCAL_TOP_K = 20
LOCAL_MIN_P = 0.0
LOCAL_REPEAT_PENALTY = 1.0
LOCAL_MAX_OUTPUT_TOKENS = 768
LOCAL_SEED = 20260706
LOCAL_THINKING_MODE = False
LOCAL_THINKING_DISABLE_MECHANISM = "transformers_tokenizer_apply_chat_template_enable_thinking_false_then_llama_cpp_completion"

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_INITIAL_K = 15
RERANK_TOP_N = 3
RERANKER_CANDIDATES = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "ncbi/MedCPT-Cross-Encoder",
]

# Confidence calibration thresholds for answer-level reliability.
CONFIDENCE_HIGH_THRESHOLD = 0.75
CONFIDENCE_MEDIUM_THRESHOLD = 0.50


def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def download_file(url, directory, filename):
    """Downloads a file from a URL if it doesn't already exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        logging.info(f"Downloading {filename} from {url}...")
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request) as response, open(filepath, "wb") as out_file:
            out_file.write(response.read())
        logging.info(f"Downloaded {filename} successfully.")
    else:
        logging.info(f"{filename} already exists. Skipping download.")
    return filepath

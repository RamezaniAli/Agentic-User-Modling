import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(__file__)

# ---------- VectorDB ----------
CHROMA_DB_PATH = BASE_DIR + "/data/chroma_db"
TOP_K_PERSONA = 2
TOP_K_INTERACTION = 3

# ---------- Evaluation ----------
EVAL_SCORE_THRESHOLD = 0.7

# ---------- LLM / Embedding Models ----------

EMBEDDING_MODEL_NAME = "text-embedding-3-small"

# LLM_MODEL_NAME = "gpt-4.1-mini"
LLM_MODEL_NAME = "gemini-2.0-flash"


BASE_URL = "https://api.avalai.ir/v1"
REQUEST_TIMEOUT = 30
MODEL_TEMP = 0.3

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

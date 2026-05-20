"""Pipeline configuration — API keys, models, paths, concurrency."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# API keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

# Models
REWRITER_MODEL = "claude-opus-4-6"
CLASSIFIER_MODEL = "claude-opus-4-6"
VERIFIER_MODEL = "gpt-5.4"

# Concurrency
MAX_CONCURRENT_TASKS = int(os.getenv("CONCURRENCY", "8"))
MAX_REFIX_ITERATIONS = 2

# PII engine toggles
ENABLE_PRESIDIO = True
ENABLE_GOOGLE_DLP = True
ENABLE_OPENAI_FILTER = True
ENABLE_LLM_CLASSIFIER = True

# GCS
GCS_BUCKET = "meta-openclaw-privacy"
GCS_SERVICE_ACCOUNT_PATH = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS",
    str(BASE_DIR.parent / "service-account.json")
)

# Personas
PERSONAS_PATH = BASE_DIR.parent / "privacy-personas.json"
TASKS_PATH = BASE_DIR.parent / "privacy-task-generator" / "outputs" / "tasks_all.json"

# Quality gates
MIN_PRIVACY_COMPLIANCE = 4
MIN_TOOL_CALLS = 3
REQUIRE_PRIVACY_DECISION_POINT = True

# Workspace
WORKSPACE_PREFIX = "/home/user/OpenClawTrainer/workspace"

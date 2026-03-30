"""
config.py
Central configuration for the portfolio agent.
Edit this file to change models, paths, or database settings.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# API
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL      = "gemini-flash-latest"
GEMINI_MAX_RETRIES = 5      # number of retries on 503 errors
GEMINI_RETRY_BACKOFF = 10.0  # base wait in seconds (multiplied by attempt number)

# Database
DB_PATH  = "portfolio.db"
DATA_DIR = "data"

# Evaluator
GROUND_TRUTH_PATH    = "ground_truth_dataset.json"
EVALUATION_OUT_PATH  = "evaluation_results.json"
AGENT_REQUEST_DELAY  = 3.0  # seconds between evaluator requests to avoid rate limiting
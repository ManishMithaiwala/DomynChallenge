"""
test_api.py
Smoke test to verify the Gemini API key is correctly configured.

Usage:
    python test_api.py
"""

import os
from config import GEMINI_API_KEY, GEMINI_MODEL


def test_api_key_loaded():
    if not GEMINI_API_KEY:
        print("[FAIL] GEMINI_API_KEY not found.")
        print("       Make sure a .env file exists in this directory with:")
        print("       GEMINI_API_KEY=your_key_here")
        return False
    print(f"[PASS] GEMINI_API_KEY loaded: {GEMINI_API_KEY[:8]}...")
    return True


def test_api_connection():
    try:
        from google import genai
    except ImportError:
        print("[FAIL] google-genai package not installed.")
        print("       Run: pip install -r requirements.txt")
        return False

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents="Reply with exactly the words: API is working",
        )
        reply = response.text.strip()
        print(f"[PASS] Gemini API responded: {reply}")
        return True
    except Exception as exc:
        print(f"[FAIL] Gemini API call failed: {exc}")
        print("       Your key may be invalid or expired.")
        print("       Generate a new one at: https://aistudio.google.com/app/apikey")
        return False


def main():
    print("Testing Gemini API configuration...\n")

    key_ok = test_api_key_loaded()
    if not key_ok:
        return

    success = test_api_connection()

    print()
    if success:
        print("All checks passed. You are ready to run the agent.")
    else:
        print("One or more checks failed. See messages above.")


if __name__ == "__main__":
    main()
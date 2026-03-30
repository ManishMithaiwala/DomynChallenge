"""
evaluator.py
Runs every question in ground_truth_dataset.json through the agent and
reports how many answers match the expected results.

Match logic
-----------
text2sql questions
  The agent's answer is compared with the result of executing the ground-truth
  SQL directly against the database.  A match is declared when the agent's
  answer contains the key values from the ground-truth result set.

exposure_calculator questions
  The agent must produce a sector breakdown. The actual sector names computed
  directly by run_exposure_tool must all appear in the agent's answer.

Usage:
    python evaluator.py
    python evaluator.py --ground-truth ground_truth_dataset.json --db portfolio.db
"""

import argparse
import json
import re
import sqlite3
import sys
import time

from config import DB_PATH, DATA_DIR, GROUND_TRUTH_PATH, EVALUATION_OUT_PATH, AGENT_REQUEST_DELAY
from db_setup import setup_database
from agent import run_agent
from tools.exposure_tool import run_exposure_tool

# -- Helpers ------------------------------------------------------------------

def _execute_sql(sql: str, conn: sqlite3.Connection):
    """Return rows as a list of tuples, or None on error."""
    try:
        cursor = conn.execute(sql)
        return cursor.fetchall()
    except sqlite3.Error as exc:
        print(f"    [WARN] Ground-truth SQL error: {exc}", file=sys.stderr)
        return None


def _normalise(value) -> str:
    """Lowercase-stripped string for fuzzy comparison."""
    return str(value).strip().lower()


def _to_numeric(text: str):
    """
    Try to parse a string as a number after stripping formatting.
    Returns a float if successful, None otherwise.
    """
    cleaned = text.strip().lstrip("$").replace(",", "").replace("%", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _numeric_forms(value: str) -> list:
    """
    Return all reasonable string representations of a numeric value
    so we can match however the agent chose to format it.
    e.g. '85000000.0' -> ['85000000.0', '85000000', '85,000,000', ...]
    """
    num = _to_numeric(value)
    if num is None:
        return [value]

    forms = set()
    forms.add(str(num))
    forms.add(str(int(num)) if num == int(num) else str(num))
    forms.add(f"{num:,.0f}")
    forms.add(f"{num:,.2f}")
    forms.add(f"{num:.0f}")
    forms.add(f"{num:.2f}")
    return list(forms)


def _value_in_answer(gt_value: str, answer: str) -> bool:
    """
    Check whether a ground-truth value appears in the agent answer,
    trying both exact string match and all numeric representations.
    """
    answer_lower = answer.lower()
    if _normalise(gt_value) in answer_lower:
        return True
    for form in _numeric_forms(gt_value):
        if form.lower() in answer_lower:
            return True
    # Last resort: compare parsed floats found anywhere in the answer
    gt_num = _to_numeric(gt_value)
    if gt_num is not None:
        candidates = re.findall(r"[\d,]+\.?\d*", answer.replace("$", ""))
        for c in candidates:
            if _to_numeric(c) == gt_num:
                return True
    return False


def _best_text_col(rows: list) -> list:
    """
    Pick the column that contains the most useful labels for matching.
    Tries every non-numeric column and returns the one whose values are
    longest on average (company names beat ticker symbols).
    """
    if not rows:
        return []
    num_cols = len(rows[0])
    best_col = None
    best_avg_len = -1
    for col_idx in range(num_cols):
        sample = [str(row[col_idx]) for row in rows if row[col_idx] is not None]
        if not sample:
            continue
        if any(_to_numeric(v) is None for v in sample):
            avg_len = sum(len(v) for v in sample) / len(sample)
            if avg_len > best_avg_len:
                best_avg_len = avg_len
                best_col = col_idx
    if best_col is None:
        best_col = 0  # all numeric, fall back to first col
    return [_normalise(row[best_col]) for row in rows]


def _single_value_from_rows(rows) -> str:
    if rows and len(rows) == 1 and len(rows[0]) == 1:
        return _normalise(rows[0][0])
    return ""


def _run_agent_with_retry(question: str, conn, verbose: bool,
                          max_retries: int = 3, backoff: float = 5.0) -> str:
    """Run the agent, retrying on 503 rate-limit errors with exponential backoff."""
    for attempt in range(1, max_retries + 1):
        answer = run_agent(question, conn, verbose=verbose)
        if "503" not in answer and "UNAVAILABLE" not in answer:
            return answer
        wait = backoff * attempt
        print(f"    [RETRY] attempt {attempt}/{max_retries} hit rate limit, "
              f"waiting {wait:.0f}s...", file=sys.stderr)
        time.sleep(wait)
    return answer  # return last answer even if still failing


def check_sql_match(agent_answer: str, ground_truth_sql: str,
                    expected_result_type: str, conn: sqlite3.Connection) -> tuple[bool, str]:
    """
    Compare agent_answer against the ground-truth SQL result.
    Returns (matched: bool, detail: str).
    """
    gt_rows = _execute_sql(ground_truth_sql, conn)
    if gt_rows is None:
        return False, "Could not execute ground-truth SQL"

    if expected_result_type == "single_value":
        gt_val = _single_value_from_rows(gt_rows)
        matched = _value_in_answer(gt_val, agent_answer)
        detail = f"expected '{gt_val}' in answer"

    elif expected_result_type == "list":
        gt_values = [_normalise(row[0]) for row in gt_rows]
        matched = all(_value_in_answer(v, agent_answer) for v in gt_values)
        detail = f"expected {len(gt_values)} items in answer"

    else:  # "table" -- find the best label column to check against the answer
        label_values = _best_text_col(gt_rows)
        matched = all(_value_in_answer(v, agent_answer) for v in label_values)
        detail = f"expected {len(gt_rows)} rows with label values in answer"

    return matched, detail


def check_exposure_match(agent_answer: str, portfolio_name: str,
                         conn: sqlite3.Connection) -> tuple[bool, str]:
    """
    Verify the agent produced a sector-exposure breakdown for the portfolio.
    Checks that every sector name returned by run_exposure_tool appears in
    the agent's answer.
    """
    direct_result = run_exposure_tool(portfolio_name, conn)

    # Extract sector names: lines that contain a % and are not headers or dividers
    sector_names = []
    for line in direct_result.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("-") or stripped.startswith("Sector") or \
                stripped.startswith("(") or stripped.startswith("TOTAL"):
            continue
        if "%" in stripped:
            sector = stripped.split()[0]
            if sector:
                sector_names.append(sector.lower())

    answer_lower = agent_answer.lower()
    sectors_found = [s for s in sector_names if s in answer_lower]
    matched = len(sectors_found) == len(sector_names) and len(sector_names) > 0
    detail = f"{len(sectors_found)}/{len(sector_names)} sectors found in answer"
    return matched, detail


# -- Main evaluator -----------------------------------------------------------

def evaluate(ground_truth_path: str, db_path: str, data_dir: str, verbose: bool):
    with open(ground_truth_path, encoding="utf-8") as f:
        dataset = json.load(f)

    conn = setup_database(db_path=db_path, data_dir=data_dir)

    questions = dataset["questions"]
    total = len(questions)
    passed = 0
    results = []

    print(f"\nEvaluating {total} questions against '{ground_truth_path}'\n")
    print("=" * 72)

    for item in questions:
        qid        = item["id"]
        qtype      = item["type"]
        difficulty = item["difficulty"]
        question   = item["question"]
        gt         = item["ground_truth"]

        print(f"Q{qid:02d} [{difficulty:6s}] {question}")

        # --- Run agent with retry on rate-limit errors ---
        try:
            agent_answer = _run_agent_with_retry(question, conn, verbose=verbose)
        except Exception as exc:
            agent_answer = f"AGENT ERROR: {exc}"

        if verbose:
            print(f"         Answer: {agent_answer[:200]}")

        # --- Check match ---
        if qtype == "text2sql":
            matched, detail = check_sql_match(
                agent_answer,
                gt["sql_query"],
                gt["expected_result_type"],
                conn,
            )
        elif qtype == "exposure_calculator":
            matched, detail = check_exposure_match(
                agent_answer,
                gt["parameters"]["portfolio_name"],
                conn,
            )
        else:
            matched, detail = False, f"Unknown question type: {qtype}"

        status = "PASS" if matched else "FAIL"
        if matched:
            passed += 1

        print(f"         {status}  ({detail})")
        print()

        results.append({
            "id": qid,
            "type": qtype,
            "difficulty": difficulty,
            "question": question,
            "passed": matched,
            "detail": detail,
            "agent_answer": agent_answer,
        })

        # Small delay to avoid rate-limiting
        time.sleep(AGENT_REQUEST_DELAY)

    # -- Summary --------------------------------------------------------------
    print("=" * 72)
    print(f"\nRESULTS: {passed}/{total} passed ({100*passed/total:.1f}%)\n")

    # Breakdown by difficulty
    for diff in ("easy", "medium", "hard"):
        subset = [r for r in results if r["difficulty"] == diff]
        if subset:
            n_pass = sum(1 for r in subset if r["passed"])
            print(f"  {diff.capitalize():6s}: {n_pass}/{len(subset)}")

    # Breakdown by type
    for qtype in ("text2sql", "exposure_calculator"):
        subset = [r for r in results if r["type"] == qtype]
        if subset:
            n_pass = sum(1 for r in subset if r["passed"])
            print(f"  {qtype}: {n_pass}/{len(subset)}")

    print()

    # Save detailed results
    with open(EVALUATION_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": {"total": total, "passed": passed}, "results": results}, f, indent=2)
    print(f"Detailed results saved to '{EVALUATION_OUT_PATH}'")

    conn.close()
    return passed, total


def main():
    parser = argparse.ArgumentParser(description="Portfolio Agent Evaluator")
    parser.add_argument("--ground-truth", default=GROUND_TRUTH_PATH)
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    evaluate(args.ground_truth, args.db, args.data_dir, args.verbose)


if __name__ == "__main__":
    main()
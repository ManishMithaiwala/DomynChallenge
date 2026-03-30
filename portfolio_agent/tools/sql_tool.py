"""
tools/sql_tool.py
Converts a natural-language question into a SQL query using Google Gemini,
executes it against the SQLite database, and returns a human-readable answer.
"""

import json
import re
import sqlite3
import time

from google import genai
from google.genai import errors as genai_errors
from config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_MAX_RETRIES, GEMINI_RETRY_BACKOFF

# -- Schema summary sent to Gemini --------------------------------------------
DB_SCHEMA_SUMMARY = """You are an expert SQL assistant for a portfolio management database.

DATABASE TABLES AND COLUMNS:

sectors(sector_id, sector_name, sector_description, industry_group)
securities(security_id, symbol, company_name, asset_type ['Stock'|'Bond'],
           sector_id, market_cap, current_price, currency, exchange, country,
           listing_date, maturity_date, coupon_rate)
benchmarks(benchmark_id, benchmark_name, benchmark_symbol, benchmark_type,
           description, inception_date)
portfolios(portfolio_id, portfolio_name, creation_date, target_risk_level,
           total_aum, strategy_type, benchmark_index, status)
holdings(holding_id, portfolio_id, security_id, quantity, purchase_price,
         purchase_date, current_weight, cost_basis)
transactions(transaction_id, portfolio_id, security_id, transaction_type,
             quantity, price, transaction_date, fees, settlement_date, notes)
historical_prices(price_id, security_id, price_date, open_price, high_price,
                  low_price, close_price, volume, adjusted_close)
portfolio_performance(performance_id, portfolio_id, performance_date, nav,
                      total_return_1m, total_return_3m, total_return_6m,
                      total_return_1y, volatility, sharpe_ratio, max_drawdown)
risk_metrics(risk_id, portfolio_id, calculation_date, var_95, var_99, cvar_95,
             beta, correlation_sp500, tracking_error, information_ratio,
             sortino_ratio)

RULES:
- Use SQLite syntax only.
- Only use columns and tables listed above. Do NOT invent column names, aliases, or metrics that are not in the schema.
- If the question uses a term like "diversification score" or "DiversificationScore", compute it from real columns (e.g. sector_count / total_holdings) and use a plain alias like diversification_ratio.
- Only SELECT data. Never use INSERT, UPDATE, DELETE, DROP, or any DDL.
- Return ONLY a valid JSON object with a single key "sql" whose value is the SQL query string.
- Do NOT include markdown fences, explanation, or any other text.
- Example: {"sql": "SELECT COUNT(*) FROM portfolios;"}
"""

# Only these SQL statement types are permitted
_ALLOWED_STATEMENTS = ("SELECT", "WITH")

# Known valid table names in the schema
_VALID_TABLES = {
    "sectors", "securities", "benchmarks", "portfolios",
    "holdings", "transactions", "historical_prices",
    "portfolio_performance", "risk_metrics",
}


def _extract_sql(raw: str) -> str | None:
    """
    Pull the SQL string out of the model response.
    Returns the SQL string, or None if nothing usable was found.
    """
    # Strip markdown fences
    cleaned = re.sub(r"```[a-z]*", "", raw).replace("```", "").strip()

    # Try JSON parse first
    try:
        return json.loads(cleaned)["sql"]
    except (json.JSONDecodeError, KeyError):
        pass

    # Match any SQL statement so _validate_sql can reject forbidden ones
    match = re.search(
        r"(WITH|SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE)\b.*",
        cleaned, re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(0).strip()

    return None


def _validate_sql(sql: str) -> str | None:
    """
    Validate that the SQL string is safe and well-formed.

    Checks:
      1. Starts with an allowed statement (SELECT or WITH).
      2. Does not contain data-modifying keywords.
      3. References at least one known table from the schema.
      4. Parses without error via SQLite's EXPLAIN (dry-run).

    Returns an error message string if invalid, or None if the SQL is clean.
    """
    normalised = sql.strip().upper()

    # 1. Must start with an allowed keyword
    if not any(normalised.startswith(kw) for kw in _ALLOWED_STATEMENTS):
        return (
            f"Query must start with one of {_ALLOWED_STATEMENTS}. "
            f"Got: '{sql[:40]}...'"
        )

    # 2. Must not contain write/DDL keywords
    forbidden = ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
                 "REPLACE", "TRUNCATE", "ATTACH", "DETACH", "PRAGMA")
    for kw in forbidden:
        # Use word-boundary check to avoid false positives inside names
        if re.search(rf"\b{kw}\b", normalised):
            return f"Query contains forbidden keyword: {kw}"

    # 3. Must reference at least one known table
    tables_referenced = {
        t for t in _VALID_TABLES
        if re.search(rf"\b{t}\b", sql, re.IGNORECASE)
    }
    if not tables_referenced:
        return f"Query does not reference any known table. Valid tables: {sorted(_VALID_TABLES)}"

    return None  # all checks passed


def _dry_run_sql(sql: str, conn: sqlite3.Connection) -> str | None:
    """
    Use SQLite's EXPLAIN to parse the query without executing it.
    Returns an error message if the query is syntactically invalid, else None.
    """
    try:
        conn.execute(f"EXPLAIN {sql}")
        return None
    except sqlite3.Error as exc:
        return f"SQL syntax error: {exc}"


def _format_results(rows: list, description: list) -> str:
    """Format sqlite rows into a readable string."""
    if not rows:
        return "Query returned no results."

    col_names = [d[0] for d in description]

    # Single value
    if len(rows) == 1 and len(col_names) == 1:
        return str(rows[0][0])

    # Table
    lines = [" | ".join(col_names)]
    lines.append("-" * max(len(lines[0]), 40))
    for row in rows:
        lines.append(" | ".join("" if v is None else str(v) for v in row))
    return "\n".join(lines)


def run_sql_tool(question: str, conn: sqlite3.Connection) -> str:
    """
    Given a natural-language question and a live DB connection,
    return a human-readable answer string.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)

    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=f"{DB_SCHEMA_SUMMARY}\n\nConvert this question to SQL:\n\n{question}",
            )
            break
        except genai_errors.ServerError:
            if attempt == GEMINI_MAX_RETRIES:
                return "Gemini is currently unavailable due to high demand. Please try again in a few minutes."
            time.sleep(GEMINI_RETRY_BACKOFF * attempt)

    raw = response.text.strip()

    # Step 1: extract SQL from model response
    sql = _extract_sql(raw)
    if sql is None:
        return "Could not extract a SQL query from the model response."

    # Step 2: validate content and safety
    validation_error = _validate_sql(sql)
    if validation_error:
        return f"Generated query failed validation: {validation_error}"

    # Step 3: dry-run parse via SQLite EXPLAIN
    syntax_error = _dry_run_sql(sql, conn)
    if syntax_error:
        return f"Generated query has a syntax error: {syntax_error}"

    # Step 4: execute and return results
    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        return _format_results(rows, cursor.description)
    except sqlite3.Error as exc:
        return f"SQL execution error: {exc}"
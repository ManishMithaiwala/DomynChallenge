"""
tools/sql_tool.py
Converts a natural-language question into a SQL query using Google Gemini,
executes it against the SQLite database, and returns a human-readable answer.
"""

import json
import re
import sqlite3

from google import genai
from config import GEMINI_API_KEY, GEMINI_MODEL

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
- Return ONLY a valid JSON object with a single key "sql" whose value is the SQL query string.
- Do NOT include markdown fences, explanation, or any other text.
- Example: {"sql": "SELECT COUNT(*) FROM portfolios;"}
"""


def _extract_sql(raw: str) -> str:
    """Pull the SQL string out of the model response robustly."""
    # Strip markdown fences
    cleaned = re.sub(r"```[a-z]*", "", raw).replace("```", "").strip()

    # Try JSON parse first
    try:
        return json.loads(cleaned)["sql"]
    except (json.JSONDecodeError, KeyError):
        pass

    # Fallback: look for a SELECT/WITH statement
    match = re.search(r"(WITH\b|SELECT\b).*", cleaned, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(0).strip()

    return cleaned  # last resort: use whatever came back


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

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"{DB_SCHEMA_SUMMARY}\n\nConvert this question to SQL:\n\n{question}",
    )

    raw = response.text.strip()
    sql = _extract_sql(raw)

    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        return _format_results(rows, cursor.description)
    except Exception as exc:
        return f"SQL execution error: {exc}\nGenerated SQL: {sql}"
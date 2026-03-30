"""
tools/exposure_tool.py
Calculates sector exposures for a named portfolio.

Logic:
  1. Fetch all equity (Stock) holdings for the portfolio.
  2. Compute each holding's market value  = quantity * current_price.
  3. Sum market values by sector.
  4. Express each sector total as a percentage of the total equity market value.
  Bonds are excluded from the calculation entirely.
"""

import sqlite3


HOLDINGS_QUERY = """
SELECT
    sec.sector_name,
    h.quantity,
    s.current_price,
    h.quantity * s.current_price AS market_value
FROM holdings h
JOIN portfolios p  ON h.portfolio_id  = p.portfolio_id
JOIN securities s  ON h.security_id   = s.security_id
JOIN sectors sec   ON s.sector_id     = sec.sector_id
WHERE p.portfolio_name = ?
  AND s.asset_type = 'Stock'
"""


def run_exposure_tool(portfolio_name: str, conn: sqlite3.Connection) -> str:
    """
    Returns a human-readable sector-exposure breakdown for *portfolio_name*.
    """
    cursor = conn.execute(HOLDINGS_QUERY, (portfolio_name,))
    rows = cursor.fetchall()

    if not rows:
        # Check whether the portfolio exists at all
        exists = conn.execute(
            "SELECT 1 FROM portfolios WHERE portfolio_name = ?", (portfolio_name,)
        ).fetchone()
        if not exists:
            return f"Portfolio '{portfolio_name}' not found in the database."
        return f"Portfolio '{portfolio_name}' has no equity holdings to calculate sector exposure."

    # Aggregate market value by sector
    sector_values: dict[str, float] = {}
    for sector_name, _qty, _price, market_value in rows:
        sector_values[sector_name] = sector_values.get(sector_name, 0.0) + market_value

    total_equity_value = sum(sector_values.values())

    # Build output
    lines = [
        f"Sector Exposure Breakdown — {portfolio_name}",
        f"(Equity holdings only; bonds excluded)",
        "",
        f"{'Sector':<28} {'Market Value ($)':>18} {'Exposure (%)':>13}",
        "-" * 62,
    ]

    for sector, value in sorted(sector_values.items(), key=lambda x: x[1], reverse=True):
        pct = (value / total_equity_value) * 100
        lines.append(f"{sector:<28} {value:>18,.2f} {pct:>12.2f}%")

    lines += [
        "-" * 62,
        f"{'TOTAL':<28} {total_equity_value:>18,.2f} {'100.00%':>13}",
    ]

    return "\n".join(lines)
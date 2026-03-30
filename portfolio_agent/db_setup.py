"""
db_setup.py
Initialises the SQLite database from the schema and loads all CSV data files.
"""

import sqlite3
import csv
import os

SCHEMA = """
CREATE TABLE IF NOT EXISTS sectors (
    sector_id INTEGER PRIMARY KEY,
    sector_name TEXT NOT NULL UNIQUE,
    sector_description TEXT,
    industry_group TEXT
);

CREATE TABLE IF NOT EXISTS securities (
    security_id INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL UNIQUE,
    company_name TEXT NOT NULL,
    asset_type TEXT NOT NULL CHECK(asset_type IN ('Stock', 'Bond')),
    sector_id INTEGER,
    market_cap REAL,
    current_price REAL,
    currency TEXT DEFAULT 'USD',
    exchange TEXT,
    country TEXT,
    listing_date DATE,
    maturity_date DATE,
    coupon_rate REAL,
    FOREIGN KEY (sector_id) REFERENCES sectors(sector_id)
);

CREATE TABLE IF NOT EXISTS benchmarks (
    benchmark_id INTEGER PRIMARY KEY,
    benchmark_name TEXT NOT NULL UNIQUE,
    benchmark_symbol TEXT,
    benchmark_type TEXT,
    description TEXT,
    inception_date DATE
);

CREATE TABLE IF NOT EXISTS portfolios (
    portfolio_id INTEGER PRIMARY KEY,
    portfolio_name TEXT NOT NULL UNIQUE,
    creation_date DATE,
    target_risk_level TEXT,
    total_aum REAL,
    strategy_type TEXT,
    benchmark_index TEXT,
    status TEXT
);

CREATE TABLE IF NOT EXISTS holdings (
    holding_id INTEGER PRIMARY KEY,
    portfolio_id INTEGER NOT NULL,
    security_id INTEGER NOT NULL,
    quantity REAL NOT NULL,
    purchase_price REAL,
    purchase_date DATE,
    current_weight REAL,
    cost_basis REAL,
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id),
    FOREIGN KEY (security_id) REFERENCES securities(security_id),
    UNIQUE(portfolio_id, security_id)
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id INTEGER PRIMARY KEY,
    portfolio_id INTEGER NOT NULL,
    security_id INTEGER NOT NULL,
    transaction_type TEXT CHECK(transaction_type IN ('BUY', 'SELL')) NOT NULL,
    quantity REAL NOT NULL,
    price REAL NOT NULL,
    transaction_date DATE NOT NULL,
    fees REAL DEFAULT 0,
    settlement_date DATE,
    notes TEXT,
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id),
    FOREIGN KEY (security_id) REFERENCES securities(security_id)
);

CREATE TABLE IF NOT EXISTS historical_prices (
    price_id INTEGER PRIMARY KEY,
    security_id INTEGER NOT NULL,
    price_date DATE NOT NULL,
    open_price REAL,
    high_price REAL,
    low_price REAL,
    close_price REAL NOT NULL,
    volume INTEGER,
    adjusted_close REAL,
    FOREIGN KEY (security_id) REFERENCES securities(security_id),
    UNIQUE(security_id, price_date)
);

CREATE TABLE IF NOT EXISTS portfolio_performance (
    performance_id INTEGER PRIMARY KEY,
    portfolio_id INTEGER NOT NULL,
    performance_date DATE NOT NULL,
    nav REAL NOT NULL,
    total_return_1m REAL,
    total_return_3m REAL,
    total_return_6m REAL,
    total_return_1y REAL,
    volatility REAL,
    sharpe_ratio REAL,
    max_drawdown REAL,
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id),
    UNIQUE(portfolio_id, performance_date)
);

CREATE TABLE IF NOT EXISTS risk_metrics (
    risk_id INTEGER PRIMARY KEY,
    portfolio_id INTEGER NOT NULL,
    calculation_date DATE NOT NULL,
    var_95 REAL,
    var_99 REAL,
    cvar_95 REAL,
    beta REAL,
    correlation_sp500 REAL,
    tracking_error REAL,
    information_ratio REAL,
    sortino_ratio REAL,
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id),
    UNIQUE(portfolio_id, calculation_date)
);
"""

CSV_TABLE_MAP = [
    ("sectors.csv",             "sectors"),
    ("benchmarks.csv",          "benchmarks"),
    ("securities.csv",          "securities"),
    ("portfolios.csv",          "portfolios"),
    ("holdings.csv",            "holdings"),
    ("transactions.csv",        "transactions"),
    ("historical_prices.csv",   "historical_prices"),
    ("portfolio_performance.csv","portfolio_performance"),
    ("risk_metrics.csv",        "risk_metrics"),
]


def _coerce(value: str):
    """Convert empty strings to None; leave everything else as a string
    (SQLite will cast to the right affinity automatically)."""
    stripped = value.strip()
    return None if stripped == "" else stripped


def setup_database(db_path: str = "portfolio.db", data_dir: str = "data") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)

    for filename, table in CSV_TABLE_MAP:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"  [WARN] {filepath} not found – skipping.")
            continue

        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [tuple(_coerce(row[col]) for col in reader.fieldnames) for row in reader]

        placeholders = ", ".join("?" * len(reader.fieldnames))
        cols = ", ".join(reader.fieldnames)
        conn.executemany(
            f"INSERT OR IGNORE INTO {table} ({cols}) VALUES ({placeholders})",
            rows,
        )

    conn.commit()
    print(f"Database ready at '{db_path}'.")
    return conn


if __name__ == "__main__":
    setup_database()
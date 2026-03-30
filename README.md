# Portfolio AI Agent

An AI agent that answers natural-language questions about portfolio data using
Google Gemini as the reasoning engine.

## Project Structure

```
portfolio_agent/
    data/                        # CSV data files (loaded into DB on first run)
    tools/
        sql_tool.py              # Tool 1: Text to SQL to result (uses Gemini)
        exposure_tool.py         # Tool 2: Sector exposure calculator (pure Python)
    config.py                    # Central configuration (model, paths, settings)
    db_setup.py                  # Build SQLite DB from CSVs
    agent.py                     # CLI agent (interactive + single-question)
    evaluator.py                 # Batch evaluator against ground truth
    ground_truth_dataset.json    # Ground truth Q&A dataset
    test_api.py                  # Smoke test for Gemini API key
    requirements.txt
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API key
Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_google_gemini_api_key_here
```
Get your key at https://aistudio.google.com/app/apikey

### 3. Test the API key

Before setting up the database or running the agent, verify your API key is
working:

```bash
python test_api.py
```

Expected output:
```
Testing Gemini API configuration...

[PASS] GEMINI_API_KEY loaded: AIzaSy...
[PASS] Gemini API responded: API is working

All checks passed. You are ready to run the agent.
```

If a check fails, the script will print the specific reason and how to fix it.

### 4. Set up the database
Place the CSV data files in the `data/` directory:
```
data/
    sectors.csv
    securities.csv
    benchmarks.csv
    portfolios.csv
    holdings.csv
    transactions.csv
    historical_prices.csv
    portfolio_performance.csv
    risk_metrics.csv
```

Then initialise the SQLite database:
```bash
python db_setup.py
```

This creates `portfolio.db` in the project root, applies the full schema
(tables + indices), and loads all CSV data. The tables are created in
dependency order so foreign key constraints are satisfied. Re-running
`db_setup.py` is safe - rows are inserted with `INSERT OR IGNORE`, so no
duplicates are created.

> **Note:** `agent.py` and `evaluator.py` both call `setup_database()` on
> startup, so running `python db_setup.py` manually is optional if you're
> going straight to using the agent. It's useful to run it standalone to
> verify the data loaded correctly before the first agent run.

## Data Model

The database represents a portfolio management system with 9 tables organised
around three core concepts: what we hold, where it is held, and how it performs.

### Core entities

**portfolios** is the top-level entity. Each portfolio has a name, strategy
type (Growth, Income, Balanced, etc.), a target risk level (Low, Medium, High),
a benchmark index, a total AUM figure, and a status (Active or Passive). The
sample data contains 13 portfolios ranging from $12M to $45M AUM.

**securities** represents every investable instrument. Each security has a
symbol, company name, asset type (Stock or Bond), a current price, and exchange
details. Stocks are linked to a sector; bonds carry additional fields for
maturity date and coupon rate. The sample data has 25 stocks and 5 bonds.

**sectors** is a lookup table that classifies stocks into groups such as
Technology, Financials, Healthcare, Energy, and Consumer Staples. Each sector
belongs to a broader industry group.

### Holdings and transactions

**holdings** is the join between portfolios and securities. Each row represents
a position: how many units a portfolio owns of a given security, the purchase
price, the current weight within the portfolio, and the total cost basis.

**transactions** records every individual buy or sell event, with the price,
quantity, fees, and settlement date. While holdings reflect the current state,
transactions capture the full history of how each position was built.

### Market data

**historical_prices** stores daily OHLCV (open, high, low, close, volume) data
plus an adjusted close for each security, keyed by security and date.

### Performance and risk

**portfolio_performance** stores a daily snapshot per portfolio including NAV,
total returns over 1 month, 3 months, 6 months and 1 year, volatility, Sharpe
ratio, and maximum drawdown.

**risk_metrics** stores calculated risk figures per portfolio per date:
Value at Risk at 95% and 99% confidence, CVaR, beta against the market,
correlation to the S&P 500, tracking error, information ratio, and Sortino ratio.

**benchmarks** is a reference table of index benchmarks (S&P 500, Bloomberg
Aggregate, NASDAQ-100, etc.) that portfolios are measured against.

### How the tables relate

A portfolio has many holdings. Each holding points to one security. Each
security belongs to one sector (for stocks) or none (for bonds). Transactions,
performance snapshots, and risk metrics all link back to a portfolio. Historical
prices link back to a security. This means you can answer questions that span
the whole chain, for example: "what percentage of a portfolio's equity value is
in the Technology sector" requires joining portfolios, holdings, securities, and
sectors together.

## Usage

### Interactive agent
```bash
python agent.py
```

### Single question
```bash
python agent.py --question "How many portfolios do we have?"
python agent.py -q "What are the sector exposures for the Tech Innovation Fund?"
python agent.py -q "Show me the top 5 holdings by cost basis in the Growth Equity Fund"
```

### Run evaluator
```bash
python evaluator.py
# Results printed to console + saved to evaluation_results.json
```

## Logging

The agent uses Python's standard `logging` module. Two levels are available:

**INFO** (default) — shows the key steps for every question:
```
[INFO] Received question: How many portfolios do we have?
[INFO] Sending question to Gemini (gemini-flash-latest) to select a tool...
[INFO] Gemini selected tool: 'sql_query'
[INFO] Running SQL tool for question: How many portfolios do we have?
[INFO] Tool completed. Sending result back to Gemini...
[INFO] Agent finished. Returning answer to user.
```

**DEBUG** — adds raw tool inputs, outputs, and Gemini internals. Enable with `--verbose`:
```bash
python agent.py --verbose -q "How many portfolios do we have?"
python evaluator.py --verbose
```

Additional debug output:
```
[DEBUG] Agentic loop step 1: calling Gemini...
[DEBUG] Gemini finish_reason: STOP
[DEBUG] Tool arguments: {'question': 'How many portfolios do we have?'}
[DEBUG] SQL tool result: 13
[DEBUG] Final answer: There are 13 portfolios in total.
```

The evaluator logs `PASS` or `FAIL` per question with the match detail, in addition to printing the summary table to the console.

## How It Works

### Tool 1 - SQL Query Tool (`tools/sql_tool.py`)
Sends the user's question and the full DB schema to Gemini (model configured in `config.py`),
which returns a JSON object `{"sql": "..."}`. Before executing, the query passes through
four validation checks:

1. **Extract** - strips markdown fences or JSON wrappers from the model response.
2. **Statement check** - rejects anything that does not start with `SELECT` or `WITH`.
3. **Forbidden keyword check** - blocks `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, and other DDL/write statements.
4. **Table check** - rejects queries that do not reference at least one known table from the schema.
5. **Dry-run** - runs `EXPLAIN <sql>` against SQLite to catch syntax errors without touching any data.

Only queries that pass all checks are executed. The SQL generation prompt explicitly instructs
Gemini to use only columns and tables from the schema and never invent metric names or aliases.

### Tool 2 - Exposure Calculator (`tools/exposure_tool.py`)
Pure Python, no LLM needed. Fetches all **equity** holdings for the portfolio
(bonds excluded), computes `quantity * current_price` per holding, groups by
sector, and returns percentage breakdowns.

### Agent (`agent.py`)
Uses **Gemini's function-calling API** in an agentic loop:
1. Sends the user question + both tool definitions to Gemini.
2. Gemini picks the right tool and provides the input parameters.
3. The chosen tool executes locally and returns a result string.
4. The result is fed back to Gemini, which produces a final human-readable answer.

### Evaluator (`evaluator.py`)
For each question in `ground_truth_dataset.json`:
- Runs the question through the agent.
- **text2sql** questions: executes the ground-truth SQL and checks that the
  agent's answer contains the same key values.
- **exposure_calculator** questions: checks that all expected sector names
  appear in the agent's answer.
- Prints PASS/FAIL per question with a summary breakdown by difficulty and type.
- Saves detailed results to `evaluation_results.json`.

### Configuration (`config.py`)
Single source of truth for all configurable values. Edit this file to change
the Gemini model, database path, data directory, or evaluator settings without
touching any other file.
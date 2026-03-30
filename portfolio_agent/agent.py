"""
agent.py
Command-line portfolio agent.

The agent uses Google Gemini's function-calling API to decide which tool to call:
  - sql_query           -- convert a question to SQL and run it
  - exposure_calculator -- compute sector exposures for a named portfolio

Usage:
    python agent.py
    python agent.py --question "How many portfolios do we have?"
    python agent.py -q "What are the sector exposures for the Tech Innovation Fund?"
"""

import argparse
import logging
import sqlite3
import sys
import time

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_MAX_RETRIES, GEMINI_RETRY_BACKOFF, DB_PATH, DATA_DIR

from db_setup import setup_database
from tools.sql_tool import run_sql_tool
from tools.exposure_tool import run_exposure_tool

# -- Logging setup ------------------------------------------------------------

logger = logging.getLogger("portfolio_agent")


def configure_logging(verbose: bool = False) -> None:
    """
    Set up logging for the agent.
    INFO level shows the key steps (tool chosen, result size, final answer).
    DEBUG level adds raw tool inputs/outputs and Gemini finish reasons.
    """
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logging.getLogger("portfolio_agent").setLevel(level)
    logging.getLogger("portfolio_agent").addHandler(handler)


# -- Tool declarations for Gemini function calling ----------------------------
SQL_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="sql_query",
            description=(
                "Convert a natural-language question about portfolio data into SQL "
                "and execute it against the database. Use this for questions about "
                "counts, lists, aggregations, rankings, filtering or joining data "
                "across portfolios, holdings, securities, sectors, transactions, "
                "performance, and risk metrics."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "question": types.Schema(
                        type=types.Type.STRING,
                        description="The natural-language question to answer with SQL.",
                    )
                },
                required=["question"],
            ),
        )
    ]
)

EXPOSURE_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="exposure_calculator",
            description=(
                "Calculate the sector exposure breakdown for a specific portfolio. "
                "Only equity (Stock) holdings are included; bonds are excluded. "
                "Returns each sector's market value and its percentage of total "
                "equity value. Use this whenever the user asks about sector "
                "exposures, sector weights, or sector allocation for a portfolio."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "portfolio_name": types.Schema(
                        type=types.Type.STRING,
                        description="The exact name of the portfolio (e.g. 'Tech Innovation Fund').",
                    )
                },
                required=["portfolio_name"],
            ),
        )
    ]
)

SYSTEM_PROMPT = (
    "You are a helpful portfolio management assistant. "
    "You have access to two tools:\n"
    "1. sql_query - for database questions (counts, lists, comparisons, rankings, etc.)\n"
    "2. exposure_calculator - for sector exposure / sector weight breakdowns of a specific portfolio.\n\n"
    "IMPORTANT: When calling sql_query, pass the user's question exactly as they asked it. "
    "Do not rephrase, add column names, or invent metric names. "
    "The sql_query tool will handle translating it into SQL.\n\n"
    "Always use a tool to answer the user's question. Do not guess or make up data. "
    "After receiving the tool result, present it clearly to the user."
)


def _dispatch_tool(name: str, args: dict, conn: sqlite3.Connection) -> str:
    """Execute the named tool and return its string result."""
    if name == "sql_query":
        logger.info("Running SQL tool for question: %s", args["question"])
        result = run_sql_tool(args["question"], conn)
        logger.debug("SQL tool result:\n%s", result)
        return result
    elif name == "exposure_calculator":
        logger.info("Running exposure calculator for portfolio: %s", args["portfolio_name"])
        result = run_exposure_tool(args["portfolio_name"], conn)
        logger.debug("Exposure tool result:\n%s", result)
        return result
    else:
        logger.warning("Unknown tool requested: %s", name)
        return f"Unknown tool: {name}"


def _call_gemini(client, history: list, verbose: bool = False):
    """
    Call Gemini with retry on 503 ServerError.
    Returns the response object or raises after all retries are exhausted.
    """
    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        try:
            return client.models.generate_content(
                model=GEMINI_MODEL,
                contents=history,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    tools=[SQL_TOOL, EXPOSURE_TOOL],
                ),
            )
        except genai_errors.ServerError:
            if attempt == GEMINI_MAX_RETRIES:
                logger.error(
                    "Gemini unavailable after %d attempts. Please try again later.",
                    GEMINI_MAX_RETRIES,
                )
                return None  # signal failure to caller
            wait = GEMINI_RETRY_BACKOFF * attempt
            logger.warning(
                "Gemini returned 503 (attempt %d/%d). Retrying in %.0fs...",
                attempt, GEMINI_MAX_RETRIES, wait,
            )
            time.sleep(wait)


def run_agent(question: str, conn: sqlite3.Connection, verbose: bool = False) -> str:
    """
    Run one turn of the agent for *question*.
    Returns the final answer string.
    """
    logger.info("Received question: %s", question)
    logger.info("Sending question to Gemini (%s) to select a tool...", GEMINI_MODEL)

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Build conversation history
    history: list[types.Content] = []

    # Initial user message
    history.append(types.Content(role="user", parts=[types.Part(text=question)]))

    step = 0

    # Agentic loop
    while True:
        step += 1
        logger.debug("Agentic loop step %d: calling Gemini...", step)

        response = _call_gemini(client, history, verbose=verbose)

        if response is None:
            return "Gemini is currently unavailable due to high demand. Please try again in a few minutes."

        candidate = response.candidates[0]
        content = candidate.content

        logger.debug("Gemini finish_reason: %s", candidate.finish_reason)

        # Append assistant turn to history
        history.append(content)

        # Collect function calls from this response
        function_calls = [p for p in content.parts if p.function_call is not None]
        text_parts = [p.text for p in content.parts if p.text]

        # No function call -- done
        if not function_calls:
            final_answer = "\n".join(text_parts) if text_parts else "(No response generated)"
            logger.info("Agent finished. Returning answer to user.")
            logger.debug("Final answer:\n%s", final_answer)
            return final_answer

        # Execute each function call and collect results
        tool_response_parts = []
        for part in function_calls:
            fc = part.function_call
            tool_name = fc.name
            tool_args = dict(fc.args)

            logger.info("Gemini selected tool: '%s'", tool_name)
            logger.debug("Tool arguments: %s", tool_args)

            result_text = _dispatch_tool(tool_name, tool_args, conn)

            logger.info("Tool completed. Sending result back to Gemini...")

            tool_response_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=tool_name,
                        response={"result": result_text},
                    )
                )
            )

        # Feed results back as a user turn
        history.append(types.Content(role="user", parts=tool_response_parts))


def interactive_loop(conn: sqlite3.Connection):
    print("Portfolio Agent ready. Type 'quit' or 'exit' to stop.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        answer = run_agent(question, conn)
        print(f"\nAgent: {answer}\n")


def main():
    parser = argparse.ArgumentParser(description="Portfolio Management AI Agent")
    parser.add_argument("--question", "-q", help="Single question to answer (non-interactive)")
    parser.add_argument("--db", default=DB_PATH, help="Path to SQLite database")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Directory containing CSV files")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging (tool inputs, raw outputs, Gemini internals)")
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)

    conn = setup_database(db_path=args.db, data_dir=args.data_dir)

    if args.question:
        answer = run_agent(args.question, conn, verbose=args.verbose)
        print(answer)
    else:
        interactive_loop(conn)

    conn.close()


if __name__ == "__main__":
    main()
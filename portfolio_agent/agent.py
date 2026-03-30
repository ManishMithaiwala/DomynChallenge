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
import sqlite3
import sys

from google import genai
from google.genai import types
from config import GEMINI_API_KEY, GEMINI_MODEL, DB_PATH, DATA_DIR

from db_setup import setup_database
from tools.sql_tool import run_sql_tool
from tools.exposure_tool import run_exposure_tool

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
    "Always use a tool to answer the user's question. Do not guess or make up data. "
    "After receiving the tool result, present it clearly to the user."
)


def _dispatch_tool(name: str, args: dict, conn: sqlite3.Connection) -> str:
    """Execute the named tool and return its string result."""
    if name == "sql_query":
        return run_sql_tool(args["question"], conn)
    elif name == "exposure_calculator":
        return run_exposure_tool(args["portfolio_name"], conn)
    else:
        return f"Unknown tool: {name}"


def run_agent(question: str, conn: sqlite3.Connection, verbose: bool = False) -> str:
    """
    Run one turn of the agent for *question*.
    Returns the final answer string.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Build conversation history
    history: list[types.Content] = []

    # Initial user message
    history.append(types.Content(role="user", parts=[types.Part(text=question)]))

    # Agentic loop
    while True:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[SQL_TOOL, EXPOSURE_TOOL],
            ),
        )

        candidate = response.candidates[0]
        content = candidate.content  # the assistant's Content object

        if verbose:
            finish = candidate.finish_reason
            print(f"  [agent] finish_reason={finish}", file=sys.stderr)

        # Append assistant turn to history
        history.append(content)

        # Collect function calls from this response
        function_calls = [p for p in content.parts if p.function_call is not None]
        text_parts = [p.text for p in content.parts if p.text]

        # No function call -- done
        if not function_calls:
            return "\n".join(text_parts) if text_parts else "(No response generated)"

        # Execute each function call and collect results
        tool_response_parts = []
        for part in function_calls:
            fc = part.function_call
            tool_name = fc.name
            tool_args = dict(fc.args)

            if verbose:
                print(f"  [agent] calling '{tool_name}' with {tool_args}", file=sys.stderr)

            result_text = _dispatch_tool(tool_name, tool_args, conn)

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
    parser.add_argument("--verbose", "-v", action="store_true", help="Show internal agent steps")
    args = parser.parse_args()

    conn = setup_database(db_path=args.db, data_dir=args.data_dir)

    if args.question:
        answer = run_agent(args.question, conn, verbose=args.verbose)
        print(answer)
    else:
        interactive_loop(conn)

    conn.close()


if __name__ == "__main__":
    main()
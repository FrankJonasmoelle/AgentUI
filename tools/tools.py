import os
import requests
from bs4 import BeautifulSoup
from langchain_experimental.tools import PythonREPLTool
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# tool: websearch
web_search = DuckDuckGoSearchRun()

# toll: python terminal
python_repl = PythonREPLTool()


@tool
def multiply_numbers(a: float, b: float) -> float:
    """Used to multiply two numbers together. Use it for math"""
    return a * b


@tool
def divide_numbers(a: float, b: float) -> float:
    """Used to divide to numbers. Use it for math"""
    return a / b


@tool
def read_webpage(url: str) -> str:
    """Useful for scraping and reading the text content of a specific webpage URL.
    Use this after searching the web if you need to read the full article."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10, verify=False)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        # Extract text and remove extra whitespace
        text = soup.get_text(separator=" ", strip=True)

        # limit the output to 3000 characters so it doesn't crash the LLM's context window
        return text[:3000]
    except Exception as e:
        return f"Error reading webpage: {e}"


@tool
def write_to_file(filename: str, content: str) -> str:
    """Useful for saving text, reports, or code to a local file.
    Provide the filename (e.g., report.txt) and the content to save."""
    try:
        # Create an 'output' folder in your project so it doesn't clutter your root directory
        os.makedirs("output", exist_ok=True)
        filepath = os.path.join("output", filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully saved document to {filepath}"
    except Exception as e:
        return f"Error writing to file: {e}"


# group tools into list so they can be passed to agent later
AGENT_TOOLS = [
    python_repl,
    web_search,
    read_webpage,
    multiply_numbers,
    divide_numbers,
    write_to_file,
]

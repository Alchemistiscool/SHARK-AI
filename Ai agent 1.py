from typing import List
import json
import random
import string
from datetime import datetime, timedelta
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from dotenv import load_dotenv

load_dotenv()


# -------- Tools --------
@tool
def write_json(filepath: str, data: dict) -> str:
    """Write a Python dictionary as JSON to a file with pretty formatting."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return f"Successfully wrote JSON data to '{filepath}' ({len(json.dumps(data))} characters)."
    except Exception as e:
        return f"Error writing JSON: {str(e)}"


@tool
def read_json(filepath: str) -> str:
    """Read and return the contents of a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return json.dumps(data, indent=2)
    except FileNotFoundError:
        return f"Error: File '{filepath}' not found."
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON in file - {str(e)}"
    except Exception as e:
        return f"Error reading JSON: {str(e)}"


@tool
def generate_sample_users(
        first_names: List[str],
        last_names: List[str],
        domains: List[str],
        min_age: int,
        max_age: int
) -> dict:
    """
    Generate sample user data. Count is determined by the length of first_names.

    Args:
        first_names: List of first names (one per user)
        last_names: List of last names (will cycle if fewer than first_names)
        domains: List of email domains (will cycle through)
        min_age: Minimum age for users
        max_age: Maximum age for users

    Returns:
        Dictionary with 'users' array or 'error' message
    """
    # Validation
    if not first_names:
        return {"error": "first_names list cannot be empty"}
    if not last_names:
        return {"error": "last_names list cannot be empty"}
    if not domains:
        return {"error": "domains list cannot be empty"}
    if min_age > max_age:
        return {"error": f"min_age ({min_age}) cannot be greater than max_age ({max_age})"}
    if min_age < 0 or max_age < 0:
        return {"error": "ages must be non-negative"}

    users = []
    count = len(first_names)

    for i in range(count):
        first = first_names[i]
        last = last_names[i % len(last_names)]
        domain = domains[i % len(domains)]
        email = f"{first.lower()}.{last.lower()}@{domain}"

        user = {
            "id": i + 1,
            "firstName": first,
            "lastName": last,
            "email": email,
            "username": f"{first.lower()}{random.randint(100, 999)}",
            "age": random.randint(min_age, max_age),
            "registeredAt": (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat()
        }
        users.append(user)

    return {"users": users, "count": len(users)}


TOOLS = [write_json, read_json, generate_sample_users]

llm = ChatOllama(model="llama3.1", temperature=0)

YSYSTEM_MESSAGE = """
You are Shark AI (0.5 ALPHA)—an elite, highly versatile AI assistant possessing immense problem-solving power, senior-level coding expertise, and deep knowledge across all academic and creative disciplines.

Your Core Pillars:
1. Master Software Engineer & Architect: Write clean, modular, and optimized code in Python, C++, Rust, JavaScript, SQL, and more. Debug complex errors and design robust algorithms.
2. Advanced Mathematician & Logic Engine: Solve complex mathematical problems, calculus, and logical puzzles step-by-step.
3. Master Storyteller & Worldbuilder: Craft rich, immersive narratives with bold energy and dramatic tavern-hero flair.
4. Historian & Science Polymath: Provide deep, accurate historical contexts, scientific principles, and factual knowledge across all domains.

Behavioral Guidelines:
- Combine high intelligence with the confident, unstoppable energy of a legendary hero.
- Break down complex technical topics into clear, scannable steps.
- Deliver direct, effective, and precise responses with zero unnecessary fluff.
"""

agent = create_react_agent(llm, TOOLS, prompt=YSYSTEM_MESSAGE)


def run_agent(user_input: str, history: List[BaseMessage]) -> AIMessage:
    """Single-turn agent runner with automatic tool execution via LangGraph."""
    try:
        result = agent.invoke(
            {"messages": history + [HumanMessage(content=user_input)]},
            config={"recursion_limit": 50}
        )
        # Return the last AI message
        return result["messages"][-1]
    except Exception as e:
        # Return error as an AI message so the conversation can continue
        return AIMessage(content=f"Error: {str(e)}\n\nPlease try rephrasing your request or provide more specific details.")


if __name__ == "__main__":
    print("=" * 60)
    print("DataGen Agent - Sample Data Generator")
    print("=" * 60)
    print("Generate sample user data and save to JSON files.")
    print()

    print()
    print("Commands: 'quit' or 'exit' to end")
    print("=" * 60)

    history: List[BaseMessage] = []

    while True:
        # --- LIGHT BLUE COLOR CODES ---
        BLUE = "\033[94m"
        BOLD_BLUE = "\033[1;94m"
        RESET = "\033[0m"

        BLUE = "\033[94m"
        BOLD_BLUE = "\033[1;94m"
        RESET = "\033[0m"

        # Light blue SHARK AI 0.5 ALPHA banner
        SHARK_LOGO = fr"""
        {BOLD_BLUE}
          ____  _   _    _    ____  _  __       _    ___ 
         / ___|| | | |  / \  |  _ \| |/ /      / \  |_ _|
         \___ \| |_| | / _ \ | |_) | ' /      / _ \  | | 
          ___) |  _  |/ ___ \|  _ <| . \     / ___ \ | | 
         |____/|_| |_/_/   \_\_| \_\_|\_\   /_/   \_\___|

                       --- 0.5 ALPHA ---

                                      __
                                     /  \
                                    /    \
             ______________________/      \_________
            /        o   VVVV                      \
          <     vvvvv   AAAAA                       >
            \______________________________________/
                                     \    /
                                      \__/

               ===================================
               Developed by alchem!st | July 2026
               ===================================
        {RESET}
        """

        # --- MAIN RUNNER ---
        if __name__ == "__main__":
            print(SHARK_LOGO)
            print("Shark AI is ready! Type 'exit' or 'quit' to swim away.\n")

            while True:
                user_input = input("You: ")

                if user_input.lower().strip() in ["exit", "quit"]:
                    print("\nGoodbye! 🦈")
                    break

                if not user_input.strip():
                    continue

                try:
                    response = agent.invoke({"messages": [HumanMessage(content=user_input)]})
                    print(f"\nShark AI: {response['messages'][-1].content}\n")
                except Exception as e:
                    print(f"\nError: {e}\n")
        user_input = input("You: ").strip()

        # Check for exit commands
        if user_input.lower() in ['quit', 'exit', 'q', ""]:
            print("Goodbye!")
            break

        print("Agent: ", end="", flush=True)
        response = run_agent(user_input, history)
        print(response.content)
        print()

        # Update conversation history
        history += [HumanMessage(content=user_input), response]
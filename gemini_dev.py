#!/usr/bin/env python
# gemini_dev.py

import os
import sys # Retained for sys.exit calls
import click
import google.generativeai as genai
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
import subprocess
import re
from pathlib import Path

# --- Configuration ---
load_dotenv()
console = Console()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
# As of late April 2025, gemini-1.5-pro-latest is a strong choice. Adjust if needed.
MODEL_NAME = "gemini-1.5-pro-latest"

# --- Gemini Model Initialization ---
if not GEMINI_API_KEY:
    console.print("[bold red]Error: GOOGLE_API_KEY not found in environment or .env file.[/bold red]")
    sys.exit(1)

try:
    genai.configure(api_key=GEMINI_API_KEY)
    # Adjust safety threshold as needed: BLOCK_NONE, BLOCK_ONLY_HIGH, BLOCK_MEDIUM_AND_ABOVE, BLOCK_LOW_AND_ABOVE
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

    # System instruction to guide the AI's behavior and response format
    system_instruction = (
        "You are 'Gemini-Dev', an expert AI coding assistant operating in a CLI environment. "
        "Prioritize clear, concise, and accurate code generation and explanations. "
        "Format code blocks appropriately using Markdown fenced code blocks (e.g., ```python ... ```). "
        "When asked to create a project structure, provide output formatted strictly as:\n"
        "FILE: path/to/your/file.ext\n"
        "```language\n"
        "# Content for the file goes here\n"
        "```\n"
        "(Repeat for each file, ensure FILE: prefix is on its own line before the code block)\n\n"
        "Assume you are interacting with a Cloud Infrastructure Engineer studying AI/Automation. "
        "Be ready to assist with Python, CloudFormation/Terraform, Dockerfiles, shell scripts, and AI/ML concepts/code."
    )

    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        safety_settings=safety_settings,
        system_instruction=system_instruction,
    )
    # Start a chat session for conversation history within a single run
    chat = model.start_chat(history=[])

except Exception as e:
    console.print(f"[bold red]Error configuring Gemini API with model {MODEL_NAME}: {e}[/bold red]")
    sys.exit(1)


# --- Helper Function for Code Extraction ---
def extract_code(text, language=None):
    """Extracts the first code block, optionally matching a language identifier."""
    pattern = r"```(\w+)?\s*\n(.*?)\n```" # Regex to find fenced code blocks
    matches = re.findall(pattern, text, re.DOTALL)

    if not matches:
        return None # No code blocks found

    if language:
        for lang, code in matches:
            if lang and lang.lower() == language.lower():
                return code.strip()
        # If specific language not found, fall back to first block
        console.print(f"[yellow]Warning: No specific '{language}' code block found. Falling back to first block if any.[/yellow]")

    return matches[0][1].strip() # Return the content of the first block


# --- CLI Command Group Definition ---
@click.group()
def cli():
    """
    Gemini-Dev: Your AI-powered CLI development assistant.

    Use commands like 'ask', 'git-commit-msg', 'create-project'.
    Run '<command> --help' for details on each command.
    """
    pass

# --- 'ask' Command ---
@cli.command()
@click.argument('prompt', type=str)
@click.option('--context-file', '-c', type=click.Path(exists=True, dir_okay=False, resolve_path=True), help='Path to a file to provide as context.')
@click.option('--output-file', '-o', type=click.Path(resolve_path=True), help='Path to save the generated code/output directly.')
@click.option('--extract-language', '-l', type=str, help='Attempt to extract code only of this language (e.g., python, bash).')
def ask(prompt, context_file, output_file, extract_language):
    """Ask Gemini a question or request code generation/explanation."""
    full_prompt = prompt
    if context_file:
        try:
            with open(context_file, 'r', encoding='utf-8') as f:
                context_content = f.read()
            # Add clear markers for context
            full_prompt = f"--- CONTEXT FROM FILE: {Path(context_file).name} ---\n```\n{context_content}\n```\n\n--- USER PROMPT ---\n{prompt}"
            console.print(f"[cyan]Using context from:[/cyan] {context_file}")
        except Exception as e:
            console.print(f"[bold red]Error reading context file {context_file}: {e}[/bold red]")
            return

    console.print(f"[yellow]Sending request to Gemini ({MODEL_NAME})...[/yellow]")
    try:
        response = chat.send_message(full_prompt)
        response_text = response.text

        console.print("\n[bold green]--- Gemini Response ---[/bold green]")
        console.print(Markdown(response_text))
        console.print("[bold green]--- End Response ---[/bold green]\n")

        # Extract and save code/output if requested
        if output_file:
            content_to_save = None
            if extract_language:
                content_to_save = extract_code(response_text, extract_language)

            if not content_to_save:
                # Fallback to first code block or entire text if no blocks/language match
                content_to_save = extract_code(response_text) or response_text

            try:
                Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(content_to_save)
                console.print(f"[bold green]Output successfully saved to:[/bold green] {output_file}")
            except Exception as e:
                console.print(f"[bold red]Error writing to output file {output_file}: {e}[/bold red]")

    except Exception as e:
        console.print(f"[bold red]Error communicating with Gemini: {e}[/bold red]")
        if hasattr(e, 'response') and hasattr(e.response, 'prompt_feedback'):
             console.print(f"[bold red]Prompt Feedback:[/bold red] {e.response.prompt_feedback}")


# --- 'git-commit-msg' Command ---
@cli.command()
@click.option('--diff-args', default='--staged', help='Arguments for git diff (default: --staged). Use "" for unstaged.')
def git_commit_msg(diff_args):
    """Suggests a Git commit message based on changes."""
    git_command = ['git', 'diff']
    if diff_args:
        git_command.extend(diff_args.split())

    console.print(f"[cyan]Running: {' '.join(git_command)}[/cyan]")
    try:
        diff_process = subprocess.run(git_command, capture_output=True, text=True, check=False, encoding='utf-8')

        if diff_process.returncode != 0:
            console.print(f"[bold red]Error running git diff (Code: {diff_process.returncode}):[/bold red]")
            console.print(diff_process.stderr)
            return

        git_diff = diff_process.stdout
        if not git_diff.strip():
            console.print(f"[yellow]No changes detected with 'git diff {diff_args}'.[/yellow]")
            return

        prompt = (
            "Based on the following git diff, please generate one or more concise and informative commit message suggestions "
            "following conventional commit standards (e.g., feat:, fix:, chore:, docs:, style:, refactor:, test:). "
            "Provide only the commit message(s), each on a new line, without any preamble or explanation."
            f"\n\n--- GIT DIFF ---\n```diff\n{git_diff}\n```\n--- END DIFF ---"
        )

        console.print(f"[yellow]Generating commit message suggestion via Gemini ({MODEL_NAME})...[/yellow]")
        try:
            response = chat.send_message(prompt)
            console.print("[bold green]Suggested Commit Message(s):[/bold green]")
            console.print(response.text.strip())
        except Exception as e:
            console.print(f"[bold red]Error communicating with Gemini: {e}[/bold red]")

    except FileNotFoundError:
         console.print("[bold red]Error: 'git' command not found. Is Git installed and in your PATH?[/bold red]")
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred during git diff: {e}[/bold red]")


# --- 'create-project' Command ---
@cli.command()
@click.argument('description', type=str)
@click.option('--output-dir', '-d', type=click.Path(file_okay=False, resolve_path=True), default='.', help='Directory to create the project in (default: current directory).')
def create_project(description, output_dir):
    """Generates a basic project structure based on a description using Gemini."""

    prompt = (
        f"Generate a basic project file structure for the following description: '{description}'.\n"
        "Strictly follow the output format specified in the system instructions: list each file to create "
        "with its relative path prefixed by 'FILE: ' on its own line, followed immediately by a Markdown code block "
        "containing the file's content. Include relevant files like source code (e.g., main.py), .gitignore, requirements.txt (if applicable), etc."
    )

    console.print(f"[yellow]Generating project structure via Gemini ({MODEL_NAME})...[/yellow]")
    try:
        response = chat.send_message(prompt)
        response_text = response.text

        console.print("\n[bold cyan]--- Proposed Project Structure (Raw Response) ---[/bold cyan]")
        console.print(Markdown(response_text))
        console.print("[bold cyan]--- End Proposed Structure ---[/bold cyan]\n")

        pattern = r"^FILE:\s*(.+?)\s*?\n```(?:\w+)?\s*\n(.*?)\n```" # Regex to find FILE: path and code block
        matches = re.findall(pattern, response_text, re.MULTILINE | re.DOTALL)

        if not matches:
            console.print("[bold red]Error: Could not parse the project structure from Gemini's response.[/bold red]")
            console.print("[yellow]Please check the raw response above. Ensure the model followed the 'FILE: path/file.ext' format.[/yellow]")
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        created_files_count = 0
        for rel_path_str, content in matches:
            rel_path = Path(rel_path_str.strip())
            abs_path = (output_path / rel_path).resolve()

            if output_path.resolve() not in abs_path.parents and abs_path != output_path.resolve():
                 console.print(f"[bold red]Security Error: Path '{rel_path}' attempts to escape the output directory '{output_path}'. Skipping.[/bold red]")
                 continue

            try:
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                with open(abs_path, 'w', encoding='utf-8') as f:
                    f.write(content.strip())
                console.print(f"[green]Created:[/green] {abs_path}")
                created_files_count += 1

            except Exception as e:
                console.print(f"[bold red]Error creating file {abs_path}: {e}[/bold red]")

        if created_files_count > 0:
            console.print(f"\n[bold green]Project structure based on '{description}' created successfully in '{output_path}'. ({created_files_count} file(s))[/bold green]")
        else:
            console.print("[yellow]No files were created based on the parsed response.[/yellow]")

    except Exception as e:
        console.print(f"[bold red]Error during project creation or communication with Gemini: {e}[/bold red]")


# --- Main Execution Guard ---
if __name__ == '__main__':
    cli()
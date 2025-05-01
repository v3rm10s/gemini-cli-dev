#!/usr/bin/env python
# gemini_dev.py

import os
import sys
import click
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.ai.generativelanguage as glm
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
import subprocess
import re
import json
from pathlib import Path
from typing import List

# --- Configuration ---
load_dotenv()
console = Console()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-2.0-pro-latest"
HISTORY_FILE = Path("chat_history.json")

# --- Gemini Model Initialization & History Loading ---
if not GEMINI_API_KEY:
    console.print("[bold red]Error: GOOGLE_API_KEY not found.[/bold red]")
    sys.exit(1)
try:
    genai.configure(api_key=GEMINI_API_KEY)
    safety_settings = { #	BLOCK_NONE, BLOCK_LOW_AND_ABOVE, BLOCK_MEDIUM_AND_ABOVE, BLOCK_ONLY_HIGH
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }
    system_instruction = (
         "You are 'Gemini-Dev', an expert AI coding assistant..."
    )
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        safety_settings=safety_settings,
        system_instruction=system_instruction,
    )
except Exception as e:
    console.print(f"[bold red]Error configuring Gemini API: {e}[/bold red]")
    sys.exit(1)

# --- History Loading/Saving Functions ---
def load_history() -> List[glm.Content]:
    """Loads chat history from the JSON file."""
    history = []
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f: history_data = json.load(f)
            for item in history_data:
                 parts = [part['text'] for part in item.get('parts', []) if 'text' in part]
                 if item.get('role') and parts:
                     history.append(glm.Content(role=item.get('role'), parts=[glm.Part(text=p) for p in parts]))
            if history: console.print(f"[dim]Loaded {len(history)//2} previous exchanges from {HISTORY_FILE}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load/parse history from {HISTORY_FILE}: {e}[/yellow]")
            history = []
    return history

def save_history(history: List[glm.Content]):
    """Saves chat history to the JSON file."""
    try:
        history_data = []
        for msg in history:
             parts_data = [{'text': part.text} for part in msg.parts if hasattr(part, 'text')]
             if msg.role and parts_data:
                  history_data.append({'role': msg.role, 'parts': parts_data})
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2)
    except Exception as e:
        console.print(f"[bold red]Error saving chat history: {e}[/bold red]")

# --- Initialize Chat Object ---
loaded_history = load_history()
chat = model.start_chat(history=loaded_history)

# --- Helper Function for Code Extraction ---
# (extract_code function remains the same)
def extract_code(text, language=None):
    """Extracts the first code block, optionally matching a language identifier."""
    pattern = r"```(\w+)?\s*\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches: return None
    if language:
        for lang, code in matches:
            if lang and lang.lower() == language.lower(): return code.strip()
        console.print(f"[yellow]Warning: No specific '{language}' code block. Falling back to first block.[/yellow]")
    return matches[0][1].strip()


# --- CLI Command Group Definition ---
@click.group()
@click.option('--clear-history', is_flag=True, help='Clear the chat history before running.')
def cli(clear_history):
    """
    Gemini-Dev: Your AI-powered CLI development assistant with history.

    Run without arguments for a simple menu or specify a command directly.
    Use --clear-history to start fresh when using direct commands.
    """
    global chat
    if clear_history:
        if HISTORY_FILE.exists():
            try:
                HISTORY_FILE.unlink()
                console.print(f"[yellow]Chat history file {HISTORY_FILE} cleared.[/yellow]")
                if 'chat' in globals(): chat.history.clear()
            except OSError as e: console.print(f"[bold red]Error clearing history file: {e}[/bold red]")
        else: console.print("[yellow]No chat history file found to clear.[/yellow]")
    pass


# --- 'ask' Command ---
@cli.command()
@click.argument('prompt', type=str)
@click.option('--context-file', '-c', type=click.Path(exists=True, dir_okay=False, resolve_path=True), help='Path to a file to provide as context.')
@click.option('--output-file', '-o', type=click.Path(resolve_path=True), help='Path to save the generated code/output directly.')
@click.option('--extract-language', '-l', type=str, help='Attempt to extract code only of this language (e.g., python, bash).')
@click.option('--generate-tests', '-t', is_flag=True, help='Request generation of unit tests (pytest) for the code.')
def ask(prompt, context_file, output_file, extract_language, generate_tests):
    """Ask Gemini a question or request code generation/explanation."""
    global chat
    full_prompt = prompt
    if context_file:
        try:
            with open(context_file, 'r', encoding='utf-8') as f: context_content = f.read()
            full_prompt = f"--- CONTEXT FROM FILE: {Path(context_file).name} ---\n```\n{context_content}\n```\n\n--- USER PROMPT ---\n{prompt}"
            console.print(f"[cyan]Using context from:[/cyan] {context_file}")
        except Exception as e: console.print(f"[bold red]Error reading context: {e}[/bold red]"); return

    if generate_tests:
        full_prompt += "\n\n--- ADDITIONAL REQUEST ---\nPlease also provide relevant unit tests for the primary code generated above, using the pytest framework..."
        console.print("[cyan]Requesting test generation (pytest)...[/cyan]")

    console.print(f"[yellow]Sending request to Gemini ({MODEL_NAME})...[/yellow]")
    try:
        response = chat.send_message(full_prompt)
        response_text = response.text
        console.print("\n[bold green]--- Gemini Response ---[/bold green]")
        console.print(Markdown(response_text))
        console.print("[bold green]--- End Response ---[/bold green]\n")
        save_history(chat.history)
        if output_file:
             content_to_save = None
             if extract_language: content_to_save = extract_code(response_text, extract_language)
             if not content_to_save: content_to_save = extract_code(response_text) or response_text
             try:
                 Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                 with open(output_file, 'w', encoding='utf-8') as f: f.write(content_to_save)
                 console.print(f"[bold green]Output saved to:[/bold green] {output_file}")
             except Exception as e: console.print(f"[bold red]Error writing file: {e}[/bold red]")
    except Exception as e:
         console.print(f"[bold red]Error communicating with Gemini: {e}[/bold red]")
         if hasattr(e, 'response') and hasattr(e.response, 'prompt_feedback'): console.print(f"[bold red]Prompt Feedback:[/bold red] {e.response.prompt_feedback}")


# --- 'git-commit-msg' Command ---
@cli.command()
@click.option('--diff-args', default='--staged', help='Arguments for git diff (default: --staged). Use "" for unstaged.')
def git_commit_msg(diff_args):
    """Suggests a Git commit message based on changes."""


# --- 'create-project' Command ---
@cli.command()
@click.argument('description', type=str)
@click.option('--output-dir', '-d', type=click.Path(file_okay=False, resolve_path=True), default='.', help='Directory to create the project in (default: current directory).')
def create_project(description, output_dir):
    """Generates a basic project structure based on a description using Gemini."""


# --- NEW: Simple Text Menu Function ---
def run_simple_menu():
    """Displays a simple text menu to choose and run commands."""
    global chat # Needed to clear history

    console.print("\n[bold cyan]Welcome to Gemini-Dev Menu![/bold cyan]")
    console.print("\nCurrent Loaded Model:",MODEL_NAME.upper())

    # Define menu items: (command_key, description)
    # command_key should match the click command function name or be a special key like 'exit'
    menu_items = [
        ("ask", "Ask Gemini (Code Gen, Explain, etc.)"),
        ("git_commit_msg", "Suggest Git Commit Message"),
        ("create_project", "Create New Project Structure"),
        ("clear_history", "Clear Chat History"),
        ("exit", "Exit")
    ]

    while True:
        print("\nPlease choose an action:")
        for i, item in enumerate(menu_items):
            print(f"  {i + 1}. {item[1]}") # Display number and description

        try:
            choice_str = input("Enter your choice (number): ").strip()
            if not choice_str.isdigit():
                print("Invalid input. Please enter a number.")
                continue

            choice_num = int(choice_str)
            if not 1 <= choice_num <= len(menu_items):
                print("Invalid choice number.")
                continue

            command_key, _ = menu_items[choice_num - 1]

            if command_key == "exit":
                break

            if command_key == "clear_history":
                confirm_clear = input(f"Are you sure you want to delete {HISTORY_FILE}? (y/n): ").lower().strip()
                if confirm_clear == 'y':
                    if HISTORY_FILE.exists():
                        try:
                            HISTORY_FILE.unlink()
                            console.print(f"[yellow]Chat history file {HISTORY_FILE} cleared.[/yellow]")
                            if 'chat' in globals(): chat.history.clear() # Clear in-memory
                        except OSError as e: console.print(f"[bold red]Error clearing history file: {e}[/bold red]")
                    else: console.print("[yellow]No chat history file found to clear.[/yellow]")
                else:
                    print("Clear history cancelled.")
                continue # Go back to menu choices

            # --- Gather arguments for the chosen command using input() ---
            kwargs = {}
            selected_command_func = cli.get_command(None, command_key)
            if not selected_command_func: # Should not happen if menu keys match commands
                 print(f"Error: Command '{command_key}' not found internally.")
                 continue

            console.print(f"\n[bold]--- Running: {command_key} ---[/bold]")

            # Iterate through the command's parameters (defined in @click decorators)
            for param in selected_command_func.params:
                if isinstance(param, click.Argument):
                    # Loop until valid input for required arguments
                    while True:
                        value = input(f"Enter value for '{param.name}': ").strip()
                        if value or not param.required: # Accept empty if not required (though args usually are)
                            kwargs[param.name] = value
                            break
                        else:
                            print(f"'{param.name}' is required.")
                elif isinstance(param, click.Option):
                    # Handle options (flags, values)
                    prompt_text = f"{param.help} Enter value for '--{param.name}' (or press Enter to skip): "
                    if param.is_flag:
                        value_str = input(f"{param.help} Use '--{param.name}'? (y/n): ").lower().strip()
                        kwargs[param.name] = (value_str == 'y')
                    else:
                        # Optional value parameters
                        value = input(prompt_text).strip()
                        if value: # Only add kwarg if user provided a value
                            # Attempt type conversion if needed (basic example)
                            try:
                                kwargs[param.name] = param.type(value) if param.type else value
                            except Exception:
                                print(f"Warning: Could not convert input '{value}' for {param.name}. Using as string.")
                                kwargs[param.name] = value
                        elif param.default is not None:
                             # If user skips, but there's a default (besides None), keep it
                             # This logic might need refinement based on desired default behavior
                             pass # Let click handle default
                        # else: value is empty, no default, don't add to kwargs

            # --- Invoke the selected Click command ---
            try:
                ctx = click.Context(selected_command_func)
                ctx.ensure_object(dict)
                # Pass False for clear_history; menu handles it separately
                ctx.params = {'clear_history': False}
                ctx.invoke(selected_command_func, **kwargs)
                console.print(f"[bold]--- Finished: {command_key} ---[/bold]")
            except Exception as invoke_err:
                 console.print(f"[bold red]Error running command '{command_key}': {invoke_err}[/bold red]")
                 import traceback; traceback.print_exc() # Show full error for debugging

            # Ask if user wants to perform another action
            continue_choice = input("\nPerform another action? (y/n): ").lower().strip()
            if continue_choice != 'y':
                break

        except (EOFError, KeyboardInterrupt):
            print("\nExiting menu.") # Handle Ctrl+D or Ctrl+C during input
            break
        except Exception as e:
            console.print(f"[bold red]An unexpected error occurred in menu: {e}[/bold red]")
            import traceback; traceback.print_exc()
            break

    console.print("[bold yellow]Exiting Gemini-Dev.[/bold yellow]")


# --- Main Execution Guard ---
if __name__ == '__main__':
    if len(sys.argv) == 1:
        # No command-line arguments provided, run the simple text menu
        run_simple_menu()
    else:
        # Command-line arguments were provided, let Click handle them
        try:
             # Pass standalone_mode=False to prevent Click from exiting on errors handled here
             # However, need to handle potential exceptions from Click itself if options are bad
             cli(standalone_mode=False)
        except click.exceptions.Abort:
             console.print("[yellow]Operation aborted.[/yellow]")
        except Exception as e:
            # Catch potential issues during Click processing (like bad args)
            console.print(f"[bold red]An error occurred during CLI processing: {e}[/bold red]")
            # Consider showing help or full traceback
            # import traceback; traceback.print_exc()
            sys.exit(1)
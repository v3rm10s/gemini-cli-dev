# gemini_dev.py
import os
import click
import google.generativeai as genai
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
import subprocess
import re
from pathlib import Path
import sys # Added for checking command-line args
from inquirer import select, text, confirm, Path as InquirerPath # Added for interactive menu
from inquirer.themes import GreenPassion # Optional: Choose a theme
from inquirer.exceptions import RequiredError, ValidationError # For input validation

# --- Configuration ---
load_dotenv()
console = Console()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-1.5-pro-latest" # Or your preferred model

# --- Gemini Model Initialization ---
# (Keep the existing initialization block from the previous version)
if not GEMINI_API_KEY:
    console.print("[bold red]Error: GOOGLE_API_KEY not found in environment or .env file.[/bold red]")
    exit(1)

try:
    genai.configure(api_key=GEMINI_API_KEY)
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
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
    chat = model.start_chat(history=[])
except Exception as e:
    console.print(f"[bold red]Error configuring Gemini API with model {MODEL_NAME}: {e}[/bold red]")
    exit(1)


# --- Helper Function for Code Extraction ---
# (Keep the existing extract_code function)
def extract_code(text, language=None):
    """Extracts the first code block, optionally matching a language."""
    pattern = r"```(\w+)?\s*\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches: return None
    if language:
        for lang, code in matches:
            if lang and lang.lower() == language.lower(): return code.strip()
    return matches[0][1].strip()


# --- CLI Command Group ---
# (Keep the existing @click.group definition)
@click.group()
def cli():
    """
    Gemini-Dev: Your AI-powered CLI development assistant.

    Run without arguments for an interactive menu or specify a command directly.
    """
    pass

# --- Ask Command ---
# (Keep the existing ask command function definition)
@cli.command()
@click.argument('prompt', type=str)
@click.option('--context-file', '-c', type=click.Path(exists=True, dir_okay=False, resolve_path=True), help='Path to a file to provide as context.')
@click.option('--output-file', '-o', type=click.Path(resolve_path=True), help='Path to save the generated code/output directly.')
@click.option('--extract-language', '-l', type=str, help='Attempt to extract code only of this language (e.g., python, bash).')
def ask(prompt, context_file, output_file, extract_language):
    """Ask Gemini a question or request code generation/explanation."""
    # --- Function Body (Keep the existing logic inside the 'ask' function) ---
    full_prompt = prompt
    if context_file:
        try:
            with open(context_file, 'r') as f: context_content = f.read()
            full_prompt = f"--- CONTEXT FROM FILE: {Path(context_file).name} ---\n```\n{context_content}\n```\n\n--- USER PROMPT ---\n{prompt}"
            console.print(f"[cyan]Using context from:[/cyan] {context_file}")
        except Exception as e:
            console.print(f"[bold red]Error reading context file {context_file}: {e}[/bold red]"); return

    console.print(f"[yellow]Sending request to Gemini ({MODEL_NAME})...[/yellow]")
    try:
        response = chat.send_message(full_prompt)
        response_text = response.text
        console.print("\n[bold green]--- Gemini Response ---[/bold green]")
        console.print(Markdown(response_text))
        console.print("[bold green]--- End Response ---[/bold green]\n")

        if output_file:
            content_to_save = None
            if extract_language:
                console.print(f"[cyan]Attempting to extract '{extract_language}' code block...[/cyan]")
                content_to_save = extract_code(response_text, extract_language)
                if not content_to_save: console.print(f"[yellow]Warning: No specific '{extract_language}' code block found. Falling back to first block/full text.[/yellow]")
            if not content_to_save: content_to_save = extract_code(response_text) or response_text
            try:
                Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w') as f: f.write(content_to_save)
                console.print(f"[bold green]Output successfully saved to:[/bold green] {output_file}")
            except Exception as e: console.print(f"[bold red]Error writing to output file {output_file}: {e}[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Error communicating with Gemini: {e}[/bold red]")
        if hasattr(e, 'response') and hasattr(e.response, 'prompt_feedback'): console.print(f"[bold red]Prompt Feedback:[/bold red] {e.response.prompt_feedback}")
    # --- End of 'ask' function body ---

# --- Git Commit Message Suggestion Command ---
# (Keep the existing git_commit_msg command function definition)
@cli.command()
@click.option('--diff-args', default='--staged', help='Arguments for git diff (default: --staged). Use "" for unstaged.')
def git_commit_msg(diff_args):
    """Suggests a Git commit message based on changes."""
    # --- Function Body (Keep the existing logic inside the 'git_commit_msg' function) ---
    git_command = ['git', 'diff']
    if diff_args: git_command.extend(diff_args.split())
    console.print(f"[cyan]Running: {' '.join(git_command)}[/cyan]")
    try:
        diff_process = subprocess.run(git_command, capture_output=True, text=True, check=False)
        if diff_process.returncode != 0:
            console.print(f"[bold red]Error running git diff (Code: {diff_process.returncode}):[/bold red]\n{diff_process.stderr}"); return
        git_diff = diff_process.stdout
        if not git_diff.strip():
            console.print(f"[yellow]No changes detected with 'git diff {diff_args}'.[/yellow]"); return
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
        except Exception as e: console.print(f"[bold red]Error communicating with Gemini: {e}[/bold red]")
    except FileNotFoundError: console.print("[bold red]Error: 'git' command not found. Is Git installed and in your PATH?[/bold red]")
    except Exception as e: console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
    # --- End of 'git_commit_msg' function body ---


# --- Create Project Structure Command ---
# (Keep the existing create_project command function definition)
@cli.command()
@click.argument('description', type=str)
@click.option('--output-dir', '-d', type=click.Path(file_okay=False, resolve_path=True), default='.', help='Directory to create the project in (default: current directory).')
def create_project(description, output_dir):
    """Generates a basic project structure based on a description."""
     # --- Function Body (Keep the existing logic inside the 'create_project' function) ---
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
        console.print("\n[bold cyan]--- Proposed Project Structure ---[/bold cyan]")
        console.print(Markdown(response_text))
        console.print("[bold cyan]--- End Proposed Structure ---[/bold cyan]\n")
        pattern = r"^FILE:\s*(.+?)\s*?\n```(?:\w+)?\s*\n(.*?)\n```"
        matches = re.findall(pattern, response_text, re.MULTILINE | re.DOTALL)
        if not matches:
            console.print("[bold red]Error: Could not parse the project structure from Gemini's response.[/bold red]")
            console.print("[yellow]Check raw response. Ensure model followed 'FILE: path/file.ext' format.[/yellow]"); return
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        created_files = []
        for rel_path_str, content in matches:
            rel_path = Path(rel_path_str.strip())
            abs_path = (output_path / rel_path).resolve()
            if output_path.resolve() not in abs_path.parents:
                 console.print(f"[bold red]Security Error: Path '{rel_path}' escapes output directory '{output_path}'. Skipping.[/bold red]"); continue
            try:
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                with open(abs_path, 'w') as f: f.write(content.strip())
                created_files.append(str(rel_path))
                console.print(f"[green]Created:[/green] {abs_path}")
            except Exception as e: console.print(f"[bold red]Error creating file {abs_path}: {e}[/bold red]")
        if created_files: console.print(f"\n[bold green]Project structure based on '{description}' created successfully in '{output_path}'.[/bold green]")
        else: console.print("[yellow]No files were created based on the response.[/yellow]")
    except Exception as e: console.print(f"[bold red]Error during project creation or communication with Gemini: {e}[/bold red]")
    # --- End of 'create_project' function body ---


# --- NEW: Interactive Menu Function ---
def run_interactive_menu():
    """Displays an interactive menu to choose and run commands."""
    console.print("[bold cyan]Welcome to Gemini-Dev Interactive Mode![/bold cyan]")

    main_menu_choices = [
        ("ask", "💬 Ask Gemini (Code Gen, Explain, etc.)"),
        ("git_commit_msg", "📝 Suggest Git Commit Message"),
        ("create_project", "🏗️ Create New Project Structure"),
        ("exit", "🚪 Exit")
    ]

    while True: # Loop until user chooses to exit
        try:
            action = select(
                message="Choose an action:",
                choices=[choice[1] for choice in main_menu_choices], # Show descriptions
                default=None,
                pointer="👉",
                # vi_mode=True # Optional: enable vi keybindings
            ).execute()

            # Find the command key associated with the selection
            command_key = None
            for key, desc in main_menu_choices:
                if desc == action:
                    command_key = key
                    break

            if command_key == "exit":
                console.print("[bold yellow]Exiting interactive mode.[/bold yellow]")
                break # Exit the loop

            if command_key == "ask":
                # --- Collect args for 'ask' ---
                prompt = text(message="Enter your prompt:", validate=lambda r: len(r) > 0, invalid_message="Prompt cannot be empty.").execute()
                context_file = InquirerPath(message="Context file path (optional, press Enter to skip):", only_files=True, exists=True).execute()
                output_file = InquirerPath(message="Output file path (optional, press Enter to skip):").execute()
                extract_language = text(message="Extract specific language code (optional, e.g., python):").execute()

                # Prepare args for click invoke
                kwargs = {'prompt': prompt}
                if context_file: kwargs['context_file'] = context_file
                if output_file: kwargs['output_file'] = output_file
                if extract_language: kwargs['extract_language'] = extract_language

            elif command_key == "git_commit_msg":
                 # --- Collect args for 'git_commit_msg' ---
                 use_staged = confirm(message="Use staged changes (--staged)?", default=True).execute()
                 diff_args_str = "--staged" if use_staged else ""
                 if not use_staged:
                     custom_args = text(message="Enter custom 'git diff' arguments (optional, e.g., 'HEAD~1'):").execute()
                     if custom_args:
                         diff_args_str = custom_args # Override if custom args provided

                 kwargs = {'diff_args': diff_args_str}

            elif command_key == "create_project":
                 # --- Collect args for 'create_project' ---
                 description = text(message="Describe the project you want to create:", validate=lambda r: len(r) > 0, invalid_message="Description cannot be empty.").execute()
                 output_dir = InquirerPath(
                     message="Output directory (default: current):",
                     only_directories=True,
                     default="."
                 ).execute()
                 kwargs = {'description': description, 'output_dir': output_dir}

            else: # Should not happen if menu is set up correctly
                console.print("[bold red]Internal Error: Unknown command key.[/bold red]")
                continue

            # --- Invoke the selected Click command ---
            try:
                # Get the Click command object
                selected_command = cli.get_command(None, command_key) # Pass None for ctx when getting command outside run
                # Create a context to invoke the command
                ctx = click.Context(selected_command)
                ctx.invoke(selected_command, **kwargs)
            except Exception as invoke_err:
                 console.print(f"[bold red]Error running command '{command_key}': {invoke_err}[/bold red]")

            # Ask if user wants to perform another action
            if not confirm(message="Perform another action?", default=True).execute():
                console.print("[bold yellow]Exiting interactive mode.[/bold yellow]")
                break # Exit the loop

        except RequiredError:
            console.print("[bold red]Input is required. Exiting interactive mode.[/bold red]")
            break
        except ValidationError:
             console.print("[bold red]Invalid input. Please try again.[/bold red]")
             # Continue the loop to let user retry
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Operation cancelled by user. Exiting interactive mode.[/bold yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]An unexpected error occurred in the interactive menu: {e}[/bold red]")
            break # Exit on unexpected errors

# --- Main Execution Guard ---
if __name__ == '__main__':
    # Check if command-line arguments were passed (sys.argv[0] is the script name)
    if len(sys.argv) == 1:
        # No arguments provided, run interactive menu
        run_interactive_menu()
    else:
        # Arguments were provided, let Click handle them
        cli()
#!usr/bin/python

# gemini_dev.py
import os
import click
import google.generativeai as genai
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

load_dotenv()
console = Console()

# Configure the Gemini API
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    # Safety settings can be adjusted here if needed
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-latest", # Or your preferred model
        safety_settings=safety_settings
        # System instructions can be added here for tuning
        # system_instruction="You are a helpful AI assistant specialized in coding...",
        )
    chat = model.start_chat(history=[]) # Start a chat session for context
except Exception as e:
    console.print(f"[bold red]Error configuring Gemini API: {e}[/bold red]")
    console.print("Ensure GOOGLE_API_KEY is set correctly in your .env file.")
    exit() # Exit if API cannot be configured

@click.group()
def cli():
    """A CLI tool enhanced with Gemini for development tasks."""
    pass

@cli.command()
@click.argument('prompt', type=str)
@click.option('--context-file', '-c', type=click.Path(exists=True), help='Path to a file to provide as context.')
@click.option('--output-file', '-o', type=click.Path(), help='Path to save the generated code directly.')
def ask(prompt, context_file, output_file):
    """Ask Gemini a question or request code generation."""
    full_prompt = prompt
    if context_file:
        try:
            with open(context_file, 'r') as f:
                context_content = f.read()
            full_prompt = f"Context from {context_file}:\n```\n{context_content}\n```\n\nUser Prompt: {prompt}"
            console.print(f"[cyan]Using context from:[/cyan] {context_file}")
        except Exception as e:
            console.print(f"[bold red]Error reading context file {context_file}: {e}[/bold red]")
            return

    console.print("[yellow]Sending request to Gemini...[/yellow]")
    try:
        # Send prompt to the chat session
        response = chat.send_message(full_prompt)

        # Display the response
        console.print(Markdown(response.text)) # Use Markdown for better formatting

        # Extract and save code if requested
        if output_file:
            # Basic code extraction (assumes code is in markdown blocks)
            # More sophisticated extraction might be needed
            code_blocks = [block.strip() for block in response.text.split('```') if block.strip() and '\n' in block] # Simple heuristic
            if code_blocks:
                # Try to get the first block, assuming it's the primary code
                # You might need logic to handle multiple blocks or specify language
                code_to_save = code_blocks[0]
                # Remove potential language identifier from the first line
                first_line_end = code_to_save.find('\n')
                if first_line_end != -1:
                     first_line = code_to_save[:first_line_end].strip()
                     # Check if the first line looks like a language identifier (e.g., python, javascript)
                     if first_line.isalnum() and len(first_line) < 15: # Simple check
                         code_to_save = code_to_save[first_line_end + 1:]


                try:
                    with open(output_file, 'w') as f:
                        f.write(code_to_save)
                    console.print(f"[bold green]Code successfully exported to:[/bold green] {output_file}")
                except Exception as e:
                    console.print(f"[bold red]Error writing to output file {output_file}: {e}[/bold red]")
            else:
                console.print("[yellow]No distinct code block found to export.[/yellow]")

    except Exception as e:
        console.print(f"[bold red]Error communicating with Gemini: {e}[/bold red]")


if __name__ == '__main__':
    cli()
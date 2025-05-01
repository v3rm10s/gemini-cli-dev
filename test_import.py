# test_import.py
#!/usr/bin/env python3
print("Attempting to import inquirerpy...")
try:
    # Try the exact import path needed by gemini_dev.py
    from InquirerPy import inquirer
    print("Import successful!")
    print(f"Found inquirerpy 'inquirer' object: {inquirer}")
    # Optional: Try finding the base package path too
    import inquirerpy
    print(f"Found inquirerpy package at: {inquirerpy.__path__}")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

import sys
print(f"\nPython Executable: {sys.executable}")
print("Sys Path (where Python looks for modules):")
for p in sys.path:
    print(f"- {p}")
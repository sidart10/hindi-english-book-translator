#!/usr/bin/env python3
"""
Demonstration of the CLI Interface
Shows various usage examples
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd):
    """Run a command and display output"""
    print(f"\n{'=' * 70}")
    print(f"Running: {cmd}")
    print('=' * 70)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Demonstrate CLI usage"""
    print("🧪 CLI DEMONSTRATION")
    print("=" * 70)
    
    # Check if service account exists
    if not Path("service-account.json").exists():
        print("⚠️  service-account.json not found. Please set up Google Cloud credentials.")
        return
    
    # 1. Show help
    print("\n1️⃣ SHOWING HELP")
    run_command("python3 src/cli.py --help")
    
    # 2. Create a sample Hindi text file
    print("\n2️⃣ CREATING SAMPLE INPUT FILE")
    sample_text = """नमस्ते! यह एक परीक्षण पुस्तक है।

इस पुस्तक में हम हिंदी से अंग्रेजी अनुवाद की क्षमताओं का परीक्षण करेंगे।

यह तीसरा वाक्य है। क्या आप इसे समझ सकते हैं?"""
    
    with open("demo_input.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)
    print("✅ Created demo_input.txt")
    
    # 3. Dry run to estimate cost
    print("\n3️⃣ DRY RUN - COST ESTIMATION")
    run_command("python3 src/cli.py --input demo_input.txt --output demo_output.docx --dry-run")
    
    # 4. Run with custom batch size
    print("\n4️⃣ CUSTOM BATCH SIZE EXAMPLE")
    run_command("python3 src/cli.py --input demo_input.txt --output demo_output.docx --batch-size 10 --dry-run")
    
    # 5. Show verbose output
    print("\n5️⃣ VERBOSE MODE EXAMPLE")
    run_command("python3 src/cli.py --input demo_input.txt --output demo_output.docx --dry-run --verbose")
    
    # Clean up
    if Path("demo_input.txt").exists():
        os.remove("demo_input.txt")
        print("\n✅ Cleaned up demo files")
    
    print("\n" + "=" * 70)
    print("✨ CLI demonstration complete!")
    print("\nTo run actual translation, use:")
    print("python3 src/cli.py --input <your-book.pdf> --output <translation.docx>")


if __name__ == "__main__":
    main() 
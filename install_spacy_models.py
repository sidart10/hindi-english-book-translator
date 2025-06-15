#!/usr/bin/env python3
"""
Install required spaCy models for Quality Assurance
"""

import subprocess
import sys

def install_spacy_models():
    """Install required spaCy language models"""
    
    models = [
        ("en_core_web_sm", "English language model"),
        ("xx_sent_ud_sm", "Multi-language sentence segmentation model")
    ]
    
    print("Installing spaCy language models for Quality Assurance...")
    print("=" * 60)
    
    for model, description in models:
        print(f"\nInstalling {model} ({description})...")
        
        try:
            # Run spacy download command
            result = subprocess.run(
                [sys.executable, "-m", "spacy", "download", model],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"✅ Successfully installed {model}")
            else:
                print(f"❌ Failed to install {model}")
                print(f"Error: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error installing {model}: {e}")
    
    print("\n" + "=" * 60)
    print("spaCy model installation complete!")
    print("\nNote: If any models failed to install, you can manually install them with:")
    print("  python -m spacy download en_core_web_sm")
    print("  python -m spacy download xx_sent_ud_sm")

if __name__ == "__main__":
    install_spacy_models() 
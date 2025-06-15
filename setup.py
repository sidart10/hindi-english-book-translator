#!/usr/bin/env python3
"""
Setup script for Hindi-English Book Translator
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="hindi-english-book-translator",
    version="1.0.0",
    author="Sid Dani",
    author_email="your.email@example.com",
    description="A powerful system for translating Hindi books to English using Google Cloud Translation API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sidart10/hindi-english-book-translator",
    packages=find_packages(),
    package_dir={'': 'src'},
    py_modules=[
        'cli',
        'main_controller',
        'translation_engine',
        'document_processor',
        'cost_meter',
        'mistral_ocr_processor'
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "click>=8.0.0",
        "google-cloud-aiplatform>=1.38.0",
        "PyMuPDF>=1.23.0",
        "pytesseract>=0.3.10",
        "Pillow>=10.0.0",
        "python-docx>=0.8.11",
        "ebooklib>=0.18",
        "aiohttp>=3.9.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
        ],
        "mistral": [
            "mistralai>=1.0.0",  # When available
        ]
    },
    entry_points={
        "console_scripts": [
            "book-translator=cli:main",
            "hindi-translator=cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.md", "*.txt"],
    },
) 
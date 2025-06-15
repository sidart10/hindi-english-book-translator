#!/usr/bin/env python3
"""
LaTeX Output Generator for Hindi-English Book Translation
Generates professional LaTeX documents suitable for Overleaf and book publishing
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import re


class LaTeXOutputGenerator:
    """Generate LaTeX output for translated books"""
    
    def __init__(self, title: str = "Hindi to English Translation", 
                 author: str = "Translation System",
                 document_class: str = "book"):
        """
        Initialize LaTeX generator
        
        Args:
            title: Book title
            author: Author name
            document_class: LaTeX document class (book, article, report)
        """
        self.title = self._escape_latex(title)
        self.author = self._escape_latex(author)
        self.document_class = document_class
        
        # LaTeX packages for multilingual support
        self.packages = [
            "\\usepackage[utf8]{inputenc}",
            "\\usepackage[T1]{fontenc}",
            "\\usepackage{babel}",
            "\\usepackage{polyglossia}",  # Better multilingual support
            "\\setdefaultlanguage{english}",
            "\\setotherlanguage{hindi}",
            "\\usepackage{fontspec}",  # For XeLaTeX/LuaLaTeX
            "\\usepackage{devanagari}",  # Hindi script support
            "\\usepackage{geometry}",
            "\\usepackage{fancyhdr}",
            "\\usepackage{hyperref}",
            "\\usepackage{color}",
            "\\usepackage{soul}",  # For highlighting
            "\\usepackage{marginnote}",  # For margin notes
            "\\usepackage{footnote}",
            "\\usepackage{array}",
            "\\usepackage{longtable}",
            "\\usepackage{booktabs}",
            "\\usepackage{microtype}",  # Better typography
            "\\usepackage{setspace}",
            "\\usepackage{parskip}",
        ]
        
        # Set up fonts for Hindi support
        self.font_setup = [
            "% Font setup for Hindi support",
            "\\newfontfamily\\devanagarifont[Script=Devanagari]{Noto Sans Devanagari}",
            "\\newfontfamily\\englishfont{Times New Roman}",
            "",
            "% Page setup",
            "\\geometry{a4paper, margin=1in}",
            "\\setstretch{1.15}",  # Line spacing
            "",
            "% Header and footer",
            "\\pagestyle{fancy}",
            "\\fancyhf{}",
            "\\fancyhead[LE,RO]{\\thepage}",
            "\\fancyhead[LO,RE]{\\leftmark}",
            "\\renewcommand{\\headrulewidth}{0.4pt}",
            "",
            "% Custom commands",
            "\\newcommand{\\qarning}[1]{\\marginnote{\\color{red}\\tiny #1}}",
            "\\definecolor{qualitylow}{RGB}{255,200,200}",
            "\\definecolor{qualitymed}{RGB}{255,255,200}",
        ]
    
    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters"""
        if not text:
            return ""
        
        # LaTeX special characters that need escaping
        special_chars = {
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '^': r'\textasciicircum{}',
            '\\': r'\textbackslash{}',
        }
        
        # Replace special characters
        for char, replacement in special_chars.items():
            text = text.replace(char, replacement)
        
        return text
    
    def generate_latex(self, translated_sentences: List[Dict], 
                      output_path: str,
                      include_source: bool = False,
                      include_qa_details: bool = True) -> str:
        """
        Generate LaTeX document from translated sentences
        
        Args:
            translated_sentences: List of sentence dictionaries with translations
            output_path: Path to save the LaTeX file
            include_source: Whether to include source Hindi text
            include_qa_details: Whether to include QA warnings
            
        Returns:
            Path to the generated LaTeX file
        """
        # Ensure output path has .tex extension
        output_path = Path(output_path)
        if output_path.suffix != '.tex':
            output_path = output_path.with_suffix('.tex')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write document header
            f.write(self._generate_header())
            
            # Begin document
            f.write("\n\\begin{document}\n\n")
            
            # Title page
            f.write(self._generate_title_page())
            
            # Table of contents (for book class)
            if self.document_class == "book":
                f.write("\\tableofcontents\n\\newpage\n\n")
            
            # Main content
            f.write(self._generate_content(translated_sentences, 
                                         include_source, 
                                         include_qa_details))
            
            # End document
            f.write("\n\\end{document}\n")
        
        print(f"   LaTeX document saved to: {output_path}")
        
        # Also generate a compilation script
        self._generate_compilation_script(output_path)
        
        return str(output_path)
    
    def _generate_header(self) -> str:
        """Generate LaTeX document header"""
        header = f"\\documentclass[12pt,a4paper]{{{self.document_class}}}\n\n"
        
        # Add packages
        header += "% Required packages\n"
        header += "\n".join(self.packages) + "\n\n"
        
        # Add font setup
        header += "\n".join(self.font_setup) + "\n\n"
        
        # Document info
        header += f"\\title{{{self.title}}}\n"
        header += f"\\author{{{self.author}}}\n"
        header += f"\\date{{\\today}}\n"
        
        return header
    
    def _generate_title_page(self) -> str:
        """Generate title page"""
        title_page = "\\maketitle\n\\newpage\n\n"
        
        # Add translation info page
        title_page += "\\section*{Translation Information}\n"
        title_page += f"\\textbf{{Translation Date:}} {datetime.now().strftime('%B %d, %Y')}\\\\\n"
        title_page += "\\textbf{Translation System:} Hindi-English Book Translator\\\\\n"
        title_page += "\\textbf{Quality Assurance:} MQM-based scoring with multilingual validation\\\\\n"
        title_page += "\n\\vspace{1cm}\n"
        title_page += "\\textbf{Note:} This document was automatically translated from Hindi to English. "
        title_page += "Quality warnings are indicated in the margins where applicable.\n"
        title_page += "\\newpage\n\n"
        
        return title_page
    
    def _generate_content(self, sentences: List[Dict], 
                         include_source: bool,
                         include_qa: bool) -> str:
        """Generate main content"""
        content = ""
        current_page = None
        current_chapter = None
        
        for i, sent in enumerate(sentences):
            # Handle page/chapter changes
            page = sent.get("page")
            
            if page != current_page:
                if current_page is not None:
                    content += "\n\\newpage\n\n"
                
                current_page = page
                
                # Add chapter heading if using book class
                if self.document_class == "book":
                    content += f"\\chapter{{Page {current_page}}}\n\n"
                else:
                    content += f"\\section{{Page {current_page}}}\n\n"
            
            # Add translation
            translation = sent.get("translation", "")
            
            # Check for quality issues
            qa_result = sent.get("qa_result", {})
            has_issues = not qa_result.get("passed", True)
            qa_score = qa_result.get("overall_score", 1.0)
            
            # Format the translation based on quality
            if has_issues and include_qa:
                # Highlight problematic translations
                if qa_score < 0.7:
                    content += "\\colorbox{qualitylow}{"
                elif qa_score < 0.85:
                    content += "\\colorbox{qualitymed}{"
                else:
                    content += "{"
                
                content += self._escape_latex(translation)
                content += "}"
                
                # Add margin note with issues
                if sent.get("quality_issues"):
                    issues_text = "; ".join(sent["quality_issues"][:2])
                    if len(sent["quality_issues"]) > 2:
                        issues_text += f" (+{len(sent['quality_issues']) - 2} more)"
                    content += f"\\qarning{{QA: {qa_score:.2f} - {self._escape_latex(issues_text)}}}"
            else:
                content += self._escape_latex(translation)
            
            # Add source text if requested
            if include_source and sent.get("text"):
                content += "\\footnote{\\texthindi{" + self._escape_latex(sent["text"]) + "}}"
            
            # Add spacing between sentences
            content += "\n\n"
            
            # Add extra spacing every 5 sentences for readability
            if (i + 1) % 5 == 0:
                content += "\\vspace{0.5em}\n\n"
        
        return content
    
    def _generate_compilation_script(self, latex_path: Path):
        """Generate a script to compile the LaTeX document"""
        script_path = latex_path.with_suffix('.sh')
        
        script_content = f"""#!/bin/bash
# Compilation script for {latex_path.name}
# This script compiles the LaTeX document to PDF

echo "Compiling {latex_path.name} to PDF..."

# Use XeLaTeX for better Unicode and font support
xelatex -interaction=nonstopmode "{latex_path.name}"

# Run twice to resolve references
xelatex -interaction=nonstopmode "{latex_path.name}"

echo "Compilation complete!"
echo "Output: {latex_path.stem}.pdf"

# Alternative: Use LuaLaTeX (comment out XeLaTeX above and uncomment below)
# lualatex -interaction=nonstopmode "{latex_path.name}"
# lualatex -interaction=nonstopmode "{latex_path.name}"

# For Overleaf: This file is ready to upload directly
echo ""
echo "For Overleaf:"
echo "1. Upload {latex_path.name} to your Overleaf project"
echo "2. Set the compiler to XeLaTeX in Overleaf settings"
echo "3. Make sure Overleaf has the required fonts (Noto Sans Devanagari)"
"""
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        print(f"   Compilation script saved to: {script_path}")
    
    def generate_quality_report_appendix(self, qa_report: Dict) -> str:
        """Generate an appendix with the QA report"""
        appendix = "\n\\appendix\n"
        appendix += "\\chapter{Translation Quality Report}\n\n"
        
        # Summary statistics
        summary = qa_report.get("summary", {})
        appendix += "\\section{Summary Statistics}\n"
        appendix += "\\begin{itemize}\n"
        appendix += f"\\item Total Segments: {summary.get('total_segments', 0)}\n"
        appendix += f"\\item Average Quality Score: {summary.get('average_score', 0):.2%}\n"
        appendix += f"\\item Total MQM Defects: {summary.get('total_defects', 0)}\n"
        appendix += f"\\item Critical Errors: {summary.get('critical_errors', 0)}\n"
        appendix += f"\\item Overall Status: {'\\textcolor{green}{PASSED}' if summary.get('passed') else '\\textcolor{red}{FAILED}'}\n"
        appendix += "\\end{itemize}\n\n"
        
        # Common issues
        if qa_report.get("common_issues"):
            appendix += "\\section{Most Common Issues}\n"
            appendix += "\\begin{enumerate}\n"
            for issue, count in qa_report["common_issues"][:10]:
                appendix += f"\\item {self._escape_latex(issue)} (occurred {count} times)\n"
            appendix += "\\end{enumerate}\n\n"
        
        # Recommendations
        if qa_report.get("recommendations"):
            appendix += "\\section{Recommendations}\n"
            appendix += "\\begin{itemize}\n"
            for rec in qa_report["recommendations"]:
                appendix += f"\\item {self._escape_latex(rec)}\n"
            appendix += "\\end{itemize}\n"
        
        return appendix 
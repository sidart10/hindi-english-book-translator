# Hindi to English Book Translation System - Product Requirements Document

## Executive Summary

### Project Overview
A comprehensive system for translating books from Hindi to English while preserving literary quality, cultural nuances, and maintaining consistency throughout the translation process.

### Key Objectives
- Translate Hindi books to English with high accuracy and literary quality
- Preserve cultural context and nuances
- Maintain consistent terminology and style throughout the book
- Support various book formats (PDF, EPUB, TXT, DOCX)
- Provide quality assurance and review mechanisms

### Success Metrics
- Translation accuracy: ~~>95% semantic accuracy~~ → **MQM-Lite ≤ 3 defects / 1,000 source words; zero critical errors**
- Processing speed: ~~50-100 pages per hour~~ → **≥ 80,000 characters per hour on NVIDIA H100 or equivalent**
- Consistency score: >90% for terminology and style
- User satisfaction: >4.5/5 rating from reviewers
- **Line-alignment delta = 0 (one source line → one target line)**
- **Glossary hit-rate ≥ 98%**

## System Architecture

### High-Level Architecture
```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Input Layer   │────▶│ Processing Layer │────▶│  Output Layer   │
│  (File Upload)  │     │  (Translation)   │     │ (Export/Review) │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       │                         │
         ▼                       ▼                         ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Format Handler  │     │   AI Pipeline    │     │ Quality Control │
│  - PDF Parser   │     │ - Preprocessing  │     │ - Review UI     │
│  - EPUB Parser  │     │ - Translation    │     │ - Feedback Loop │
│  - Text Extract │     │ - Post-process   │     │ - Export Engine │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### Translation LLM Integration
```python
from google.cloud import aiplatform

def translate_chunk(lines: list[str]) -> list[str]:
    client = aiplatform.gapic.PredictionServiceClient(
        client_options={"api_endpoint": "us-central1-aiplatform.googleapis.com"}
    )
    endpoint = (
        "projects/${PROJECT_ID}/locations/us-central1/"
        "publishers/google/models/cloud-translate-text"
    )
    inst = [{
        "model": "projects/${PROJECT_ID}/locations/us-central1/models/general/translation-llm",
        "source_language_code": "hi",
        "target_language_code": "en",
        "contents": lines,
        "mimeType": "text/plain"
    }]
    resp = client.predict(instances=inst, endpoint=endpoint)
    return [t["translatedText"] for t in resp.predictions[0]["translations"]]
```

## Technical Implementation

### 1. Core Translation Engine

```python
# translation_engine.py
import asyncio
from typing import List, Dict, Tuple
~~import openai~~
~~from anthropic import Anthropic~~
from google.cloud import aiplatform
import json
import re
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TranslationConfig:
    """Configuration for translation parameters"""
    source_language: str = "hi"
    target_language: str = "en"
    model: str = ~~"claude-3-opus-20240229"~~ → **"projects/${PROJECT_ID}/locations/us-central1/models/general/translation-llm"**
    temperature: float = ~~0.3~~ → **0**
    max_characters: int = ~~4000~~ → **9000**  # Character limit (Google's limit is 10k, using 9k for headroom)
    preserve_formatting: bool = True
    ~~cultural_adaptation: str = "moderate"  # minimal, moderate, extensive~~
    style_guide: Dict = None
    glossary: Dict = None

class TranslationEngine:
    """Main translation engine with ~~multi-model support~~ → **Google Translation LLM**"""
    
    def __init__(self, config: TranslationConfig):
        self.config = config
        ~~self.anthropic = Anthropic()~~
        self.client = aiplatform.gapic.PredictionServiceClient(
            client_options={"api_endpoint": "us-central1-aiplatform.googleapis.com"}
        )
        self.endpoint = "projects/${PROJECT_ID}/locations/us-central1/publishers/google/models/cloud-translate-text"
        self.translation_cache = {}
        self.glossary = self.load_glossary()
        self.style_guide = self.load_style_guide()
        
    def load_glossary(self) -> Dict:
        """Load domain-specific terminology glossary"""
        if self.config.glossary:
            return self.config.glossary
        
        # Default glossary for Hindi-English translation
        return {
            "संस्कृति": "culture/sanskriti",
            "धर्म": "dharma/religion",
            "कर्म": "karma/action",
            # Add more terms as needed
        }
    
    def load_style_guide(self) -> Dict:
        """Load style preferences for translation"""
        if self.config.style_guide:
            return self.config.style_guide
            
        return {
            "formality": "neutral",
            "preserve_metaphors": True,
            "explain_cultural_references": True,
            "footnote_threshold": "high",
            **"no_em_dashes": True**
        }
    
    async def translate_chunk(self, lines: List[str]) -> Dict:
        """Translate a chunk of text ~~with context awareness~~"""
        
        # **Pre-flight PII scrub**
        lines = [self._scrub_pii(line) for line in lines]
        
        # Check cache first
        cache_key = f"{lines[0][:50] if lines else ''}_{len(lines)}"
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        ~~prompt = self._build_translation_prompt(text, context)~~
        
        try:
            # **Ensure we don't exceed limits: ≤ 256 sentences OR 9,000 characters**
            if len(lines) > 256:
                lines = lines[:256]
            
            total_chars = sum(len(line) for line in lines)
            if total_chars > self.config.max_characters:
                # Truncate to fit character limit
                truncated_lines = []
                char_count = 0
                for line in lines:
                    if char_count + len(line) > self.config.max_characters:
                        break
                    truncated_lines.append(line)
                    char_count += len(line)
                lines = truncated_lines
            
            translations = await self._call_translation_api(lines)
            
            results = []
            for i, (original, translation) in enumerate(zip(lines, translations)):
                result = {
                    "original": original,
                    "translation": translation,
                    "confidence": 0.95,  # Placeholder as Translation LLM doesn't provide confidence
                    "timestamp": datetime.now().isoformat()
                }
                results.append(result)
            
            # Cache the results
            self.translation_cache[cache_key] = results
            
            return results
            
        except Exception as e:
            return [{
                "original": line,
                "translation": "",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            } for line in lines]
    
    def _scrub_pii(self, text: str) -> str:
        """Remove PII before sending to LLM"""
        # Phone numbers
        text = re.sub(r'\+?[0-9]{10,15}', '[PHONE]', text)
        # Email addresses
        text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]', text)
        return text
    
    ~~def _build_translation_prompt(self, text: str, context: str) -> str:~~
        ~~"""Build comprehensive translation prompt"""~~
        # Removed - Translation LLM doesn't use prompts
    
    async def _call_translation_api(self, lines: List[str]) -> List[str]:
        """Call the Translation LLM API"""
        
        instances = [{
            "model": self.config.model,
            "source_language_code": self.config.source_language,
            "target_language_code": self.config.target_language,
            "contents": lines,
            "mimeType": "text/plain"
        }]
        
        response = self.client.predict(
            instances=instances,
            endpoint=self.endpoint,
            parameters={
                "temperature": self.config.temperature,
                "candidate_count": 1,
                "max_output_tokens": self.config.max_characters
            }
        )
        
        return [t["translatedText"] for t in response.predictions[0]["translations"]]
```

### 2. Document Processing Pipeline

```python
# document_processor.py
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from ebooklib import epub
import docx
from typing import List, Generator
import re

class DocumentProcessor:
    """Handles various document formats and extracts text"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.epub', '.txt', '.docx']
        self.page_cache = {}
        
    def process_document(self, file_path: str) -> Generator[Dict, None, None]:
        """Process document and yield ~~chunks~~ → **sentences** for translation"""
        
        file_extension = file_path.lower().split('.')[-1]
        
        if file_extension == 'pdf':
            yield from self._process_pdf(file_path)
        elif file_extension == 'epub':
            yield from self._process_epub(file_path)
        elif file_extension == 'txt':
            yield from self._process_text(file_path)
        elif file_extension == 'docx':
            yield from self._process_docx(file_path)
        else:
            raise ValueError(f"Unsupported format: {file_extension}")
    
    def _process_pdf(self, file_path: str) -> Generator[Dict, None, None]:
        """Extract text from PDF with OCR support for scanned pages"""
        
        doc = fitz.open(file_path)
        
        for page_num, page in enumerate(doc):
            # Try text extraction first
            text = page.get_text()
            
            # If no text found, try OCR
            if len(text.strip()) < 50:
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(img, lang='hin')
            
            # Split into ~~paragraphs~~ → **sentences**
            sentences = self._split_into_sentences(text)
            
            for sent_idx, sentence in enumerate(sentences):
                if sentence.strip():
                    yield {
                        "page": page_num + 1,
                        "sentence": sent_idx + 1,
                        "text": sentence,
                        "type": "body",
                        "metadata": {
                            "source": "pdf",
                            "ocr_used": len(page.get_text().strip()) < 50
                        }
                    }
        
        doc.close()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for JSONL format"""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Replace em-dashes with en-dashes (per style guide)
        text = text.replace('—', '–')
        
        # Split on sentence boundaries (Hindi uses । as sentence terminator)
        sentences = re.split(r'([।\.\?!]+)', text)
        
        # Clean and filter sentences, preserving delimiters
        cleaned_sentences = []
        delimiter = None
        for i, sent in enumerate(sentences):
            if i % 2 == 0:  # Text part
                sent = sent.strip()
                if len(sent) > 10:  # Minimum sentence length
                    cleaned_sentences.append({
                        "text": sent,
                        "delimiter": delimiter
                    })
            else:  # Delimiter part
                delimiter = sent
        
        return [s["text"] for s in cleaned_sentences]
```

### 3. Translation Memory and Consistency Manager

```python
# translation_memory.py
import sqlite3
from typing import List, Dict, Optional
import json
from datetime import datetime
import hashlib

class TranslationMemory:
    """Manages translation memory for consistency"""
    
    def __init__(self, db_path: str = "translation_memory.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_db()
        
    def _init_db(self):
        """Initialize database schema"""
        
        cursor = self.conn.cursor()
        
        # Translation memory table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS translations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_hash TEXT UNIQUE,
                source_text TEXT,
                translated_text TEXT,
                confidence REAL,
                context TEXT,
                project_id TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                usage_count INTEGER DEFAULT 1,
                **log_prob REAL**
            )
        """)
        
        # Terminology table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS terminology (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_term TEXT,
                target_term TEXT,
                context TEXT,
                notes TEXT,
                project_id TEXT,
                created_at TIMESTAMP
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_source_hash ON translations(source_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_project ON translations(project_id)")
        
        self.conn.commit()
```

### 4. Quality Assurance System

```python
# quality_assurance.py
from typing import List, Dict, Tuple
import re
from difflib import SequenceMatcher
import spacy
from collections import Counter
**import numpy as np**
**from sentence_transformers import SentenceTransformer**

class QualityAssurance:
    """Quality checks for translations"""
    
    def __init__(self):
        # Load spaCy models for both languages
        self.nlp_en = spacy.load("en_core_web_sm")
        self.nlp_hi = spacy.load("xx_sent_ud_sm")  # Multi-language sentence detection
        **self.nllb_model = SentenceTransformer('facebook/nllb-200-distilled-600M')**
        
        self.checks = [
            self.check_length_ratio,
            self.check_number_consistency,
            self.check_proper_nouns,
            self.check_punctuation,
            self.check_terminology_consistency,
            self.check_sentence_structure,
            **self.check_back_translation_similarity**,
            **self.check_glossary_hit_rate**
        ]
    
    def run_quality_checks(self, source: str, translation: str,
                          glossary: Dict = None) -> Dict:
        """Run all quality checks on translation"""
        
        results = {
            "overall_score": 1.0,
            "issues": [],
            "warnings": [],
            "suggestions": [],
            **"mqm_defects": 0**,
            **"critical_errors": 0**
        }
        
        for check in self.checks:
            check_result = check(source, translation, glossary)
            results["overall_score"] *= check_result["score"]
            
            if check_result["issues"]:
                results["issues"].extend(check_result["issues"])
                **results["mqm_defects"] += len(check_result["issues"])**
            if check_result.get("warnings"):
                results["warnings"].extend(check_result["warnings"])
            if check_result.get("suggestions"):
                results["suggestions"].extend(check_result["suggestions"])
            **if check_result.get("critical"):
                results["critical_errors"] += 1**
        
        return results
    
    **def check_back_translation_similarity(self, source: str, translation: str,
                                         glossary: Dict = None) -> Dict:
        """Check semantic similarity using back-translation"""
        
        # Back-translate English to Hindi using Translation LLM
        back_translation_instances = [{
            "model": "projects/${PROJECT_ID}/locations/us-central1/models/general/translation-llm",
            "source_language_code": "en",
            "target_language_code": "hi",
            "contents": [translation],
            "mimeType": "text/plain"
        }]
        
        try:
            back_trans_response = self.client.predict(
                instances=back_translation_instances,
                endpoint=self.endpoint
            )
            back_translated_hindi = back_trans_response.predictions[0]["translations"][0]["translatedText"]
            
            # Now compare source Hindi with back-translated Hindi using embeddings
            source_emb = self.nllb_model.encode(source, convert_to_tensor=True)
            back_trans_emb = self.nllb_model.encode(back_translated_hindi, convert_to_tensor=True)
            
            # Calculate cosine similarity
            cosine_sim = np.dot(source_emb, back_trans_emb) / (
                np.linalg.norm(source_emb) * np.linalg.norm(back_trans_emb)
            )
            
            result = {"score": 1.0, "issues": [], "warnings": []}
            
            if cosine_sim < 0.85:
                result["score"] = cosine_sim
                result["warnings"].append(
                    f"Back-translation similarity low: {cosine_sim:.2f}"
                )
            
            return result
            
        except Exception as e:
            return {"score": 0.5, "issues": [f"Back-translation failed: {str(e)}"], "warnings": []}**
    
    **def check_glossary_hit_rate(self, source: str, translation: str,
                               glossary: Dict = None) -> Dict:
        """Ensure glossary terms are used correctly"""
        
        if not glossary:
            return {"score": 1.0, "issues": []}
        
        result = {"score": 1.0, "issues": []}
        
        hits = 0
        expected = 0
        
        for term, expected_translation in glossary.items():
            if term in source:
                expected += 1
                # Check if any variant of expected translation appears
                variants = expected_translation.split('/')
                if any(variant in translation for variant in variants):
                    hits += 1
                else:
                    result["issues"].append(
                        f"Glossary term '{term}' not translated as expected"
                    )
        
        if expected > 0:
            hit_rate = hits / expected
            if hit_rate < 0.98:
                result["score"] = hit_rate
                result["issues"].append(
                    f"Glossary hit rate: {hit_rate:.2%} (expected ≥98%)"
                )
        
        return result**
```

### 5. Main Application Controller

```python
# main_controller.py
import asyncio
from pathlib import Path
from typing import List, Dict
import json
from datetime import datetime
**from asyncio import Semaphore**
**import aiohttp**

class BookTranslationController:
    """Main controller for book translation workflow"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = self.load_config(config_path)
        self.translation_engine = TranslationEngine(
            TranslationConfig(**self.config["translation"])
        )
        self.document_processor = DocumentProcessor()
        self.translation_memory = TranslationMemory()
        self.quality_assurance = QualityAssurance()
        **self.semaphore = Semaphore(self.config["processing"]["quota_limit"])**
        **self.cost_meter = CostMeter(self.config.get("monthly_budget", 1000))**
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from file"""
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    async def translate_book(self, input_path: str, output_path: str,
                           project_id: str = None) -> Dict:
        """Main method to translate entire book"""
        
        if not project_id:
            project_id = f"book_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        results = {
            "project_id": project_id,
            "input_file": input_path,
            "output_file": output_path,
            "start_time": datetime.now().isoformat(),
            "chapters": [],
            "statistics": {
                "total_words": 0,
                "translated_words": 0,
                "quality_score": 0.0,
                "errors": [],
                **"mqm_defects": 0**,
                **"critical_errors": 0**,
                **"glossary_hit_rate": 0.0**
            }
        }
        
        try:
            # Process document into **sentences**
            sentences = list(self.document_processor.process_document(input_path))
            total_sentences = len(sentences)
            
            # **Generate segmentation JSONL**
            self._generate_segmentation_jsonl(sentences, project_id)
            
            # Translate sentences in batches
            translated_sentences = []
            batch_size = 50  # Process 50 sentences at a time
            
            for i in range(0, total_sentences, batch_size):
                batch = sentences[i:i+batch_size]
                print(f"Translating sentences {i+1}-{min(i+batch_size, total_sentences)}/{total_sentences}")
                
                # **Check cost meter**
                if self.cost_meter.check_threshold(0.7):
                    print(f"WARNING: 70% of monthly budget used (${self.cost_meter.current_cost:.2f}/${self.cost_meter.budget})")
                
                # Extract text from sentence objects
                batch_texts = [s["text"] for s in batch]
                
                # Check translation memory first
                translations = []
                for text in batch_texts:
                    cached = self.translation_memory.get_translation(text, project_id)
                    if cached and cached["confidence"] > 0.9:
                        translations.append(cached["translation"])
                    else:
                        translations.append(None)
                
                # Translate missing sentences
                to_translate = [(i, text) for i, (text, trans) in enumerate(zip(batch_texts, translations)) if trans is None]
                
                if to_translate:
                    **async with self.semaphore:**
                        try:
                            indices, texts = zip(*to_translate)
                            translated = await self.translation_engine.translate_chunk(list(texts))
                            
                            # Update translations list
                            for idx, trans_obj in zip(indices, translated):
                                translations[idx] = trans_obj["translation"]
                                
                                # Save to translation memory
                                self.translation_memory.save_translation(
                                    batch_texts[idx],
                                    trans_obj["translation"],
                                    trans_obj.get("confidence", 0.95),
                                    "",
                                    project_id
                                )
                        except aiohttp.ClientResponseError as e:
                            if e.status == 429:
                                print("Rate limit hit, waiting...")
                                await asyncio.sleep(60)
                            elif e.status >= 500:
                                # Retry with exponential backoff
                                for retry in range(3):
                                    await asyncio.sleep(2 ** retry)
                                    try:
                                        translated = await self.translation_engine.translate_chunk(list(texts))
                                        break
                                    except:
                                        if retry == 2:
                                            raise
                
                # Run quality checks and combine results
                for sent_obj, translation in zip(batch, translations):
                    qa_result = self.quality_assurance.run_quality_checks(
                        sent_obj["text"],
                        translation,
                        self.translation_engine.glossary
                    )
                    
                    sent_obj["translation"] = translation
                    sent_obj["quality"] = qa_result
                    translated_sentences.append(sent_obj)
                    
                    # Update statistics
                    results["statistics"]["total_words"] += len(sent_obj["text"].split())
                    results["statistics"]["translated_words"] += len(translation.split())
                    **results["statistics"]["mqm_defects"] += qa_result["mqm_defects"]**
                    **results["statistics"]["critical_errors"] += qa_result["critical_errors"]**
                
                # **Update cost**
                total_chars = sum(len(t) for t in translations)
                self.cost_meter.add_cost(total_chars / 1000 * 0.20)  # $0.20 per 1000 chars (Google's billing)
            
            # Generate output
            self._generate_output(translated_sentences, output_path)
            
            # Calculate final statistics
            results["end_time"] = datetime.now().isoformat()
            results["statistics"]["quality_score"] = sum(
                s["quality"]["overall_score"] for s in translated_sentences
            ) / len(translated_sentences)
            
            # **Calculate glossary hit rate**
            total_hits = sum(1 for s in translated_sentences 
                           if "Glossary hit rate" not in str(s["quality"].get("issues", [])))
            results["statistics"]["glossary_hit_rate"] = total_hits / len(translated_sentences)
            
            return results
            
        except Exception as e:
            results["statistics"]["errors"].append(str(e))
            return results
    
    def _generate_segmentation_jsonl(self, sentences: List[Dict], project_id: str):
        """Generate JSONL file with segmented sentences"""
        
        jsonl_path = f"segmentation_{project_id}.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for sent in sentences:
                f.write(json.dumps(sent, ensure_ascii=False) + '\n')
        
        print(f"Segmentation JSONL saved to {jsonl_path}")
    
    def _generate_output(self, translated_sentences: List[Dict], output_path: str):
        """Generate output files (DOCX and optionally bilingual PDF)"""
        
        # Basic DOCX generation
        from docx import Document
        doc = Document()
        
        # Add title
        doc.add_heading('Translated Book', 0)
        
        current_page = None
        for sent in translated_sentences:
            # Add page breaks when page changes
            if sent.get("page") != current_page:
                if current_page is not None:
                    doc.add_page_break()
                current_page = sent.get("page")
                doc.add_heading(f'Page {current_page}', level=1)
            
            # Add translated text
            doc.add_paragraph(sent["translation"])
            
            # Add quality warnings if any
            if sent["quality"]["overall_score"] < 0.8:
                doc.add_paragraph(
                    f"⚠️ Quality Score: {sent['quality']['overall_score']:.2f}",
                    style='Caption'
                )
        
        doc.save(output_path)
        print(f"Translation saved to {output_path}")

class CostMeter:
    """Track translation costs against monthly budget"""
    
    def __init__(self, monthly_budget: float):
        self.budget = monthly_budget
        self.current_cost = 0.0
        self.cost_log = []
    
    def check_threshold(self, ratio: float) -> bool:
        """Check if cost exceeds given ratio of budget"""
        return (self.current_cost / self.budget) >= ratio
    
    def add_cost(self, cost: float):
        """Add cost and log timestamp"""
        self.current_cost += cost
        self.cost_log.append({
            "timestamp": datetime.now().isoformat(),
            "cost": cost,
            "total": self.current_cost
        })
```

~~### 6. Code Components~~

~~Polish pass is **removed** model output is considered final for fluency. If later stylistic tweaks are needed, they occur in the human QA phase.~~

### 6. Reviewer UX

```python
# qa_dashboard.py
import streamlit as st
import pandas as pd
import json
from pathlib import Path

class QADashboard:
    """**Streamlit QA dashboard for review**"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.translation_memory = TranslationMemory()
        
    def run(self):
        st.title("Translation QA Dashboard")
        
        # Load translations
        translations = self.load_translations()
        
        # Display controls
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            filter_option = st.selectbox("Filter by", ["All", "Low confidence", "Has issues"])
        with col2:
            search = st.text_input("Search text")
        
        # Filter translations
        filtered = self.filter_translations(translations, filter_option, search)
        
        # Display translations
        for idx, trans in enumerate(filtered):
            with st.container():
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Hindi (Source)**")
                    st.text(trans["original"])
                
                with col2:
                    st.markdown("**English (Translation)**")
                    
                    # Editable translation
                    edited = st.text_area(
                        "Translation",
                        value=trans["translation"],
                        key=f"trans_{idx}",
                        height=100
                    )
                    
                    # Accept/Fix toggle
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("✓ Accept", key=f"accept_{idx}"):
                            st.success("Accepted")
                    
                    with col_b:
                        if st.button("✏️ Fix", key=f"fix_{idx}"):
                            # Push edit back to TM
                            self.translation_memory.save_translation(
                                trans["original"],
                                edited,
                                1.0,  # Human-verified confidence
                                "",
                                self.project_id
                            )
                            st.success("Updated in TM")
                
                # Show quality issues
                if trans.get("quality", {}).get("issues"):
                    st.warning(f"Issues: {', '.join(trans['quality']['issues'])}")
                
                st.divider()
```

## ~~Prompt Templates for Translation~~

~~Master Translation Prompt - REMOVED: Translation LLM doesn't use prompts~~

## Configuration Files

### config.json
```json
{
  "translation": {
    "source_language": "hi",
    "target_language": "en",
    "model": "projects/${PROJECT_ID}/locations/us-central1/models/general/translation-llm",
    "temperature": 0,
    "max_characters": 9000,
    "preserve_formatting": true
  },
  "processing": {
    "chunk_size": 256,
    "batch_size": 50,
    "parallel_workers": 10,
    "quota_limit": 10
  },
  "quality": {
    "min_confidence": 0.85,
    "require_review_below": 0.7,
    "max_length_ratio": 1.5,
    "min_length_ratio": 0.8,
    **"max_mqm_defects_per_1000": 3**,
    **"max_critical_errors": 0**
  },
  "output": {
    "format": "docx",
    "include_original": false,
    "include_notes": true,
    "include_alternatives": false,
    "bilingual_pdf": true
  },
  **"quota_limit": 10**,
  **"monthly_budget": 1000**
}
```

### resources/glossaries/hi_en_general.yaml
```yaml
# Hindi-English Translation Glossary
terms:
  संस्कृति: "culture/sanskriti" 
  धर्म: "dharma/religion"
  कर्म: "karma/action"
  श्री: "Shri"
  जी: "-ji (honorific)"
  
# Proper nouns
names:
  राम: "Ram"
  कृष्ण: "Krishna"
  गंगा: "Ganga"
  
# Cultural terms requiring explanation
cultural:
  पूजा: "puja (worship ritual)"
  आरती: "aarti (offering of light)"
```

### style_guide.md
```markdown
# Translation Style Guide

## General Principles
- Maintain formal register for narrative
- Use contemporary English idioms
- Preserve cultural markers with explanations

## Formatting
- **No em-dashes (—)**, use en-dashes (–) or commas
- Preserve paragraph breaks
- Maintain sentence-level alignment

## Honorifics
- Retain Hindi honorifics with suffix notation
  - श्री → Shri
  - जी → -ji
- Do not translate titles (Pandit, Swami, etc.)

## Numbers and Dates
- Use Western numerals
- Convert lakhs/crores to millions/billions with note
```

### requirements.txt
```
~~anthropic>=0.3.0~~
~~openai>=1.0.0~~
google-cloud-aiplatform>=1.38.0
PyMuPDF>=1.23.0
ebooklib>=0.18
python-docx>=0.8.11
pytesseract>=0.3.10
Pillow>=10.0.0
spacy>=3.7.0
sentence-transformers>=2.2.0
sqlite3
asyncio
aiofiles>=23.0.0
aiohttp>=3.9.0
tqdm>=4.65.0
pandas>=2.0.0
numpy>=1.24.0
streamlit>=1.28.0  # For UI
fastapi>=0.104.0   # For API
uvicorn>=0.24.0
pydantic>=2.0.0
pytest>=7.4.0      # For testing
**ruff>=0.1.0**      # For linting
black>=23.0.0      # For formatting
pydocstyle>=6.3.0  # For docstring linting
pyyaml>=6.0        # For config files
```

## CI/CD Configuration

### .github/workflows/ci.yml
```yaml
name: CI Pipeline

on:
  pull_request:
    branches: [main, develop]
  push:
    branches: [main]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        python -m spacy download en_core_web_sm
        python -m spacy download xx_sent_ud_sm
    
    - name: Run ruff
      run: ruff check .
    
    - name: Run black
      run: black --check .
    
    - name: Run docstring linting
      run: pydocstyle .
    
    - name: Run pytest
      run: pytest tests/ -v --cov=.
```

### 7. CLI Interface

```python
# cli.py
import click
import asyncio
from pathlib import Path
import os
from google.oauth2 import service_account

@click.command()
@click.option('--input', '-i', required=True, help='Input book file (PDF/EPUB/TXT/DOCX)')
@click.option('--output', '-o', required=True, help='Output DOCX file path')
@click.option('--config', '-c', default='config.json', help='Config file path')
@click.option('--project-id', '-p', help='Google Cloud Project ID')
@click.option('--service-account', '-s', help='Path to service account JSON key')
def translate_book(input, output, config, project_id, service_account):
    """Translate a book from Hindi to English"""
    
    # Validate input file
    if not Path(input).exists():
        click.echo(f"Error: Input file '{input}' not found", err=True)
        return
    
    # Set up Google Cloud authentication
    if service_account:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account
    elif not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        click.echo("Error: No Google Cloud credentials found. Use --service-account or set GOOGLE_APPLICATION_CREDENTIALS", err=True)
        return
    
    # Set project ID in config
    if project_id:
        os.environ['GOOGLE_CLOUD_PROJECT'] = project_id
    
    # Run translation
    click.echo(f"Starting translation of '{input}'...")
    controller = BookTranslationController(config)
    
    try:
        result = asyncio.run(controller.translate_book(input, output))
        
        # Display results
        click.echo("\n✅ Translation Complete!")
        click.echo(f"Total sentences: {result['statistics']['total_words'] // 10}")  # Rough estimate
        click.echo(f"Quality score: {result['statistics']['quality_score']:.2%}")
        click.echo(f"MQM defects: {result['statistics']['mqm_defects']}")
        click.echo(f"Cost: ${controller.cost_meter.current_cost:.2f}")
        click.echo(f"Output saved to: {output}")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise

if __name__ == '__main__':
    translate_book()
```

## Rapid Testing Setup (2-Hour Implementation)

### Quick Start Guide

#### 1. Environment Setup (15 minutes)
```bash
# Clone repo and setup
git clone <repo-url>
cd book-translator
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install core dependencies only
pip install google-cloud-aiplatform PyMuPDF python-docx click aiohttp

# Download spaCy model
python -m spacy download xx_sent_ud_sm
```

#### 2. Google Cloud Setup (15 minutes)
```bash
# Install gcloud CLI
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Create service account
gcloud iam service-accounts create book-translator \
    --display-name="Book Translator Service"

# Download key
gcloud iam service-accounts keys create service-account.json \
    --iam-account=book-translator@YOUR_PROJECT_ID.iam.gserviceaccount.com

# Grant permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:book-translator@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"
```

#### 3. Minimal Test File (test_sample.py)
```python
# test_sample.py - Quick integration test
import asyncio
from translation_engine import TranslationEngine, TranslationConfig

async def test_basic_translation():
    """Test basic Hindi to English translation"""
    
    config = TranslationConfig(
        model="projects/YOUR_PROJECT_ID/locations/us-central1/models/general/translation-llm"
    )
    
    engine = TranslationEngine(config)
    
    # Test sentences
    test_lines = [
        "नमस्ते, मेरा नाम राम है।",
        "मैं भारत से हूं।",
        "यह एक परीक्षण है।"
    ]
    
    results = await engine.translate_chunk(test_lines)
    
    for result in results:
        print(f"Hindi: {result['original']}")
        print(f"English: {result['translation']}")
        print(f"Confidence: {result['confidence']}")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(test_basic_translation())
```

#### 4. Quick Test Script (30 minutes coding)
```bash
# Create minimal test book (1-2 pages)
echo "यह एक परीक्षण पुस्तक है।
इसमें कुछ सरल वाक्य हैं।
हम अनुवाद की गुणवत्ता की जांच करेंगे।" > test_book.txt

# Run translation
python cli.py \
    --input test_book.txt \
    --output translated_book.docx \
    --project-id YOUR_PROJECT_ID \
    --service-account service-account.json
```

### Minimal Implementation Checklist (1 hour)

1. **Core Files Only**:
   - [ ] `translation_engine.py` (simplified, no caching)
   - [ ] `document_processor.py` (TXT support only for testing)
   - [ ] `cli.py` (basic command interface)
   - [ ] `config.json` (minimal configuration)

2. **Skip for Now**:
   - Translation memory (use dict cache)
   - Quality assurance (basic length check only)
   - Streamlit UI (test via CLI)
   - PDF/EPUB support (start with TXT)

3. **Essential Error Handling**:
   ```python
   # Add to TranslationEngine.__init__
   def __init__(self, config: TranslationConfig):
       self.config = config
       
       # Validate credentials
       try:
           from google.cloud import aiplatform
           aiplatform.init(project=os.environ.get('GOOGLE_CLOUD_PROJECT'))
       except Exception as e:
           raise ValueError(f"Failed to initialize Google Cloud: {e}")
   ```

## Milestones (2-Hour Sprint)

- **T + 30min** – Environment setup + Google Cloud auth working
- **T + 1h** – Basic translation of test file (5-10 sentences) working
- **T + 1.5h** – TXT file processing + DOCX output functional
- **T + 2h** – End-to-end test with 1-page Hindi text complete

## Minimal Viable Implementation Structure

```
book-translator/
├── src/
│   ├── translation_engine.py    # Core translation (200 lines)
│   ├── document_processor.py    # TXT processing only (50 lines)
│   ├── cli.py                   # Command interface (80 lines)
│   └── config.json              # Configuration
├── tests/
│   ├── test_sample.py           # Basic integration test
│   └── test_book.txt            # Sample Hindi text
├── requirements-minimal.txt     # Core dependencies only
├── service-account.json         # Google Cloud credentials
└── README.md                    # Quick start guide
```

### requirements-minimal.txt (For 2-hour sprint)
```
google-cloud-aiplatform>=1.38.0
python-docx>=0.8.11
click>=8.0.0
aiohttp>=3.9.0
PyMuPDF>=1.23.0  # For PDF support
```

## Minimal PDF Processing Implementation

```python
# document_processor.py - PDF support priority
import fitz  # PyMuPDF
from typing import List, Dict

class DocumentProcessor:
    """Minimal PDF text extraction"""
    
    def process_pdf(self, file_path: str) -> List[str]:
        """Extract text from PDF - simple version"""
        
        doc = fitz.open(file_path)
        all_sentences = []
        
        for page_num, page in enumerate(doc):
            # Extract text from page
            text = page.get_text()
            
            # Simple sentence splitting
            # Split on Hindi sentence terminator (।) and common English punctuation
            import re
            sentences = re.split(r'[।\.\?!]+', text)
            
            for sent in sentences:
                sent = sent.strip()
                if len(sent) > 10:  # Skip very short fragments
                    all_sentences.append({
                        "text": sent,
                        "page": page_num + 1
                    })
        
        doc.close()
        return all_sentences

# Quick test for PDF extraction
if __name__ == "__main__":
    processor = DocumentProcessor()
    sentences = processor.process_pdf("test.pdf")
    print(f"Extracted {len(sentences)} sentences")
    for i, sent in enumerate(sentences[:5]):  # Show first 5
        print(f"{i+1}. {sent['text'][:50]}...")
```

### Updated Quick Test Script (For PDF)
```bash
# Test with your PDF directly
python cli.py \
    --input your_hindi_book.pdf \
    --output translated_book.docx \
    --project-id YOUR_PROJECT_ID \
    --service-account service-account.json
```

## Risk & Mitigation Table

| Risk | Impact | Mitigation |
|------|--------|------------|
| Translation LLM unavailable | High | Fallback to IndicTrans2 model |
| Sentence boundary detection fails | Medium | Manual sentence splitting + list-length guard |
| Cost exceeds budget | High | Real-time cost meter with 70% alert threshold |
| API rate limiting (429) | Medium | Semaphore-based concurrency control |
| Character limit exceeded | Low | Pre-chunk validation to 30k chars |
| PII exposure | High | Regex-based PII scrubbing pre-API |

## Cost Optimization

- **Ceiling: $0.03 per target-word**
- Alert when cumulative spend ≥ 70% of monthly cap
- Batch processing to minimize API calls
- Translation memory to avoid re-translating

## Security Measures

- **Pre-flight PII scrub** for phone numbers and emails
- No storage of raw user data
- Encrypted translation memory database
- API keys stored in environment variables
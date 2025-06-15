#!/usr/bin/env python3
"""
Mistral OCR Processor for Hindi Book Translation
Uses Mistral's advanced OCR API for accurate text extraction
"""

import os
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional
import base64
from dataclasses import dataclass
import asyncio
import aiohttp


@dataclass
class MistralOCRConfig:
    """Configuration for Mistral OCR"""
    api_key: str
    api_endpoint: str = "https://api.mistral.ai/v1/ocr"  # Placeholder - actual endpoint TBD
    batch_mode: bool = True  # Use batch inference for 50% cost savings
    output_format: str = "json"  # or "markdown"
    language_hints: List[str] = None  # ["hi", "en"] for Hindi-English mixed docs
    
    def __post_init__(self):
        if self.language_hints is None:
            self.language_hints = ["hi", "en"]


class MistralOCRProcessor:
    """Process documents using Mistral's OCR API"""
    
    def __init__(self, config: MistralOCRConfig):
        self.config = config
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def process_pdf(self, pdf_path: str) -> Dict:
        """
        Process PDF using Mistral OCR
        
        Returns:
            Dict with extracted text, metadata, and structure
        """
        print(f"Processing PDF with Mistral OCR: {pdf_path}")
        
        # Read PDF file
        with open(pdf_path, 'rb') as f:
            pdf_data = f.read()
            pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
        
        # Prepare request
        headers = {
            'Authorization': f'Bearer {self.config.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'document': {
                'content': pdf_base64,
                'mime_type': 'application/pdf'
            },
            'options': {
                'language_hints': self.config.language_hints,
                'output_format': self.config.output_format,
                'preserve_structure': True,
                'batch_mode': self.config.batch_mode
            }
        }
        
        try:
            # Make API request
            async with self.session.post(
                self.config.api_endpoint,
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return self._process_ocr_result(result)
                else:
                    error_text = await response.text()
                    raise Exception(f"Mistral OCR API error: {response.status} - {error_text}")
                    
        except Exception as e:
            print(f"Error processing with Mistral OCR: {e}")
            raise
    
    def _process_ocr_result(self, result: Dict) -> Dict:
        """Process and structure the OCR result"""
        
        # Extract pages and their content
        pages = []
        total_text = ""
        
        for page_data in result.get('pages', []):
            page = {
                'page_number': page_data.get('page_number'),
                'text': page_data.get('text', ''),
                'confidence': page_data.get('confidence', 0),
                'language': page_data.get('detected_language', 'unknown'),
                'paragraphs': [],
                'tables': [],
                'equations': []
            }
            
            # Extract structured elements
            for element in page_data.get('elements', []):
                if element['type'] == 'paragraph':
                    page['paragraphs'].append({
                        'text': element['text'],
                        'confidence': element.get('confidence', 0)
                    })
                elif element['type'] == 'table':
                    page['tables'].append(element)
                elif element['type'] == 'equation':
                    page['equations'].append(element)
            
            pages.append(page)
            total_text += page['text'] + "\n\n"
        
        # Calculate overall confidence
        avg_confidence = sum(p['confidence'] for p in pages) / len(pages) if pages else 0
        
        return {
            'pages': pages,
            'total_pages': len(pages),
            'full_text': total_text,
            'average_confidence': avg_confidence,
            'metadata': result.get('metadata', {}),
            'language_distribution': result.get('language_distribution', {})
        }
    
    async def compare_with_current_ocr(self, pdf_path: str, current_text: str) -> Dict:
        """Compare Mistral OCR output with current OCR text"""
        
        # Process with Mistral
        mistral_result = await self.process_pdf(pdf_path)
        mistral_text = mistral_result['full_text']
        
        # Find differences
        differences = self._find_ocr_improvements(current_text, mistral_text)
        
        return {
            'mistral_result': mistral_result,
            'improvements': differences,
            'confidence_gain': mistral_result['average_confidence']
        }
    
    def _find_ocr_improvements(self, old_text: str, new_text: str) -> Dict:
        """Identify improvements in OCR quality"""
        
        improvements = {
            'vowel_marks_fixed': 0,
            'word_separation_fixed': 0,
            'special_characters_fixed': 0,
            'examples': []
        }
        
        # Common Hindi OCR fixes
        fixes = {
            'परचय': 'परिचय',
            'बहार': 'बिहार', 
            'जमे': 'जन्मे',
            'पछले': 'पिछले',
            'वशट': 'विशिष्ट',
            'साहय': 'साहित्य',
            'संह': 'संग्रह'
        }
        
        for wrong, correct in fixes.items():
            if wrong in old_text and correct in new_text:
                improvements['vowel_marks_fixed'] += 1
                improvements['examples'].append(f"{wrong} → {correct}")
        
        # Check for better word separation
        if 'मलरहाहै' in old_text and 'मिल रहा है' in new_text:
            improvements['word_separation_fixed'] += 1
            improvements['examples'].append("मलरहाहै → मिल रहा है")
        
        return improvements


# Integration with existing document processor
class EnhancedDocumentProcessor:
    """Enhanced document processor using Mistral OCR"""
    
    def __init__(self, mistral_api_key: Optional[str] = None):
        self.use_mistral = mistral_api_key is not None
        
        if self.use_mistral:
            self.ocr_config = MistralOCRConfig(
                api_key=mistral_api_key,
                batch_mode=True,  # Save 50% on costs
                language_hints=["hi", "en"]
            )
            print("✅ Mistral OCR enabled for high-quality text extraction")
        else:
            print("⚠️  Mistral OCR not configured, using fallback OCR")
    
    async def process_document(self, file_path: str) -> Dict:
        """Process document with Mistral OCR if available"""
        
        if self.use_mistral and file_path.lower().endswith('.pdf'):
            async with MistralOCRProcessor(self.ocr_config) as processor:
                result = await processor.process_pdf(file_path)
                
                # Convert to our standard format
                sentences = []
                for page in result['pages']:
                    # Split into sentences
                    page_sentences = self._split_into_sentences(page['text'])
                    for idx, sent in enumerate(page_sentences):
                        sentences.append({
                            'page': page['page_number'],
                            'sentence': idx + 1,
                            'text': sent,
                            'confidence': page['confidence']
                        })
                
                return {
                    'sentences': sentences,
                    'ocr_quality': result['average_confidence'],
                    'ocr_engine': 'mistral'
                }
        else:
            # Fallback to existing OCR
            from document_processor import DocumentProcessor
            processor = DocumentProcessor()
            return {
                'sentences': list(processor.process_document(file_path)),
                'ocr_quality': 0.7,  # Estimate
                'ocr_engine': 'pytesseract'
            }
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        sentences = re.split(r'[।\.\?!]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]


# Example usage
async def test_mistral_ocr():
    """Test Mistral OCR with a sample PDF"""
    
    # Note: Actual API key would be needed
    api_key = os.environ.get('MISTRAL_API_KEY')
    
    if not api_key:
        print("⚠️  MISTRAL_API_KEY not set. To use Mistral OCR:")
        print("1. Sign up at https://mistral.ai")
        print("2. Get your API key from la Plateforme")
        print("3. Set MISTRAL_API_KEY environment variable")
        return
    
    processor = EnhancedDocumentProcessor(mistral_api_key=api_key)
    
    # Process a sample PDF
    result = await processor.process_document("sample.pdf")
    
    print(f"\nOCR Engine: {result['ocr_engine']}")
    print(f"OCR Quality: {result['ocr_quality']:.2%}")
    print(f"Total sentences: {len(result['sentences'])}")
    
    # Show first few sentences
    print("\nFirst 3 sentences:")
    for sent in result['sentences'][:3]:
        print(f"- Page {sent['page']}: {sent['text']}")


if __name__ == "__main__":
    asyncio.run(test_mistral_ocr()) 
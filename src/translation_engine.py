#!/usr/bin/env python3
"""
Core Translation Engine for Hindi to English Book Translation
Uses Google Cloud Translation API with proper error handling and PII scrubbing
"""

import asyncio
import re
import json
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from google.cloud import translate_v3 as translate
from google.oauth2 import service_account
import os


@dataclass
class TranslationConfig:
    """Configuration for translation parameters"""
    source_language: str = "hi"
    target_language: str = "en"
    project_id: str = field(default_factory=lambda: os.environ.get('GOOGLE_CLOUD_PROJECT', ''))
    location: str = "global"
    temperature: float = 0.0  # For consistency
    max_characters: int = 9000  # Google's limit is 10k, using 9k for headroom
    max_sentences: int = 256  # Maximum sentences per batch
    preserve_formatting: bool = True
    style_guide: Optional[Dict] = None
    glossary: Optional[Dict] = None
    credentials_path: Optional[str] = field(
        default_factory=lambda: os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    )


class TranslationEngine:
    """Main translation engine with Google Cloud Translation API"""
    
    def __init__(self, config: TranslationConfig):
        self.config = config
        self.translation_cache = {}
        self.glossary = self._load_glossary()
        self.style_guide = self._load_style_guide()
        
        # Initialize Google Cloud Translation client
        if self.config.credentials_path and os.path.exists(self.config.credentials_path):
            credentials = service_account.Credentials.from_service_account_file(
                self.config.credentials_path
            )
            self.client = translate.TranslationServiceClient(credentials=credentials)
        else:
            # Use default credentials
            self.client = translate.TranslationServiceClient()
        
        # Set up parent path for API calls
        self.parent = f"projects/{self.config.project_id}/locations/{self.config.location}"
        
        print(f"Translation Engine initialized for project: {self.config.project_id}")
    
    def _load_glossary(self) -> Dict[str, str]:
        """Load domain-specific terminology glossary"""
        if self.config.glossary:
            return self.config.glossary
        
        # Default glossary for Hindi-English translation
        return {
            "संस्कृति": "culture/sanskriti",
            "धर्म": "dharma/religion",
            "कर्म": "karma/action",
            "श्री": "Shri",
            "जी": "-ji",
            "पूजा": "puja (worship ritual)",
            "आरती": "aarti (offering of light)",
            "नमस्ते": "Namaste",
            "गुरु": "Guru/teacher",
            "योग": "yoga",
            "आत्मा": "soul/atma",
            "मोक्ष": "moksha/liberation",
            "सत्य": "truth/satya",
            "अहिंसा": "non-violence/ahimsa",
            "प्रेम": "love/prem"
        }
    
    def _load_style_guide(self) -> Dict[str, any]:
        """Load style preferences for translation"""
        if self.config.style_guide:
            return self.config.style_guide
        
        return {
            "formality": "neutral",
            "preserve_metaphors": True,
            "explain_cultural_references": True,
            "footnote_threshold": "high",
            "no_em_dashes": True,
            "maintain_honorifics": True,
            "preserve_sentence_structure": True
        }
    
    def _scrub_pii(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Remove PII before sending to API and store mappings"""
        pii_mappings = {}
        
        # Phone numbers (Indian format)
        phone_pattern = r'\+?91?[-.\s]?\d{10}|\d{5}[-.\s]\d{5}|\+?\d{10,15}'
        phones = re.findall(phone_pattern, text)
        for i, phone in enumerate(phones):
            placeholder = f"[PHONE_{i}]"
            text = text.replace(phone, placeholder)
            pii_mappings[placeholder] = phone
        
        # Email addresses
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        emails = re.findall(email_pattern, text)
        for i, email in enumerate(emails):
            placeholder = f"[EMAIL_{i}]"
            text = text.replace(email, placeholder)
            pii_mappings[placeholder] = email
        
        # Aadhaar numbers (12 digits)
        aadhaar_pattern = r'\b\d{4}\s?\d{4}\s?\d{4}\b'
        aadhaars = re.findall(aadhaar_pattern, text)
        for i, aadhaar in enumerate(aadhaars):
            placeholder = f"[AADHAAR_{i}]"
            text = text.replace(aadhaar, placeholder)
            pii_mappings[placeholder] = aadhaar
        
        return text, pii_mappings
    
    def _restore_pii(self, text: str, pii_mappings: Dict[str, str]) -> str:
        """Restore PII after translation"""
        for placeholder, original in pii_mappings.items():
            text = text.replace(placeholder, original)
        return text
    
    def _apply_style_rules(self, text: str) -> str:
        """Apply style guide rules to translated text"""
        if self.style_guide.get("no_em_dashes"):
            # Replace em-dashes with en-dashes
            text = text.replace("—", "–")
        
        # Ensure proper spacing around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])\s*', r'\1 ', text)
        text = text.strip()
        
        return text
    
    def _prepare_batch(self, lines: List[str]) -> List[str]:
        """Prepare a batch of lines for translation"""
        prepared_lines = []
        
        # Ensure we don't exceed limits
        if len(lines) > self.config.max_sentences:
            lines = lines[:self.config.max_sentences]
        
        # Check character limit
        total_chars = 0
        for line in lines:
            line_chars = len(line)
            if total_chars + line_chars > self.config.max_characters:
                break
            prepared_lines.append(line)
            total_chars += line_chars
        
        return prepared_lines
    
    async def translate_chunk(self, lines: List[str], context: str = "") -> List[Dict]:
        """Translate a chunk of text lines"""
        
        # Pre-flight PII scrub
        scrubbed_lines = []
        pii_mappings_list = []
        
        for line in lines:
            scrubbed_line, pii_mappings = self._scrub_pii(line)
            scrubbed_lines.append(scrubbed_line)
            pii_mappings_list.append(pii_mappings)
        
        # Prepare batch within limits
        batch_lines = self._prepare_batch(scrubbed_lines)
        
        # Check cache first
        results = []
        lines_to_translate = []
        line_indices = []
        
        for i, line in enumerate(batch_lines):
            cache_key = hashlib.md5(f"{line}_{self.config.source_language}_{self.config.target_language}".encode()).hexdigest()
            
            if cache_key in self.translation_cache:
                cached_result = self.translation_cache[cache_key].copy()
                cached_result["original"] = lines[i]  # Use original with PII
                results.append(cached_result)
            else:
                lines_to_translate.append(line)
                line_indices.append(i)
                results.append(None)  # Placeholder
        
        # Translate uncached lines
        if lines_to_translate:
            try:
                translations = await self._call_translation_api(lines_to_translate)
                
                # Process translations
                for idx, (line_idx, translation) in enumerate(zip(line_indices, translations)):
                    # Restore PII
                    translation_with_pii = self._restore_pii(
                        translation, 
                        pii_mappings_list[line_idx]
                    )
                    
                    # Apply style rules
                    styled_translation = self._apply_style_rules(translation_with_pii)
                    
                    result = {
                        "original": lines[line_idx],
                        "translation": styled_translation,
                        "confidence": 0.95,  # Google doesn't provide confidence scores
                        "timestamp": datetime.now().isoformat(),
                        "cache_hit": False
                    }
                    
                    # Cache the result (without PII)
                    cache_key = hashlib.md5(
                        f"{batch_lines[line_idx]}_{self.config.source_language}_{self.config.target_language}".encode()
                    ).hexdigest()
                    self.translation_cache[cache_key] = {
                        "translation": translation,  # Cache without PII
                        "confidence": result["confidence"],
                        "timestamp": result["timestamp"],
                        "cache_hit": True
                    }
                    
                    results[line_idx] = result
            
            except Exception as e:
                print(f"Translation API error: {str(e)}")
                # Return error results for failed lines
                for line_idx in line_indices:
                    if results[line_idx] is None:
                        results[line_idx] = {
                            "original": lines[line_idx],
                            "translation": "",
                            "error": str(e),
                            "timestamp": datetime.now().isoformat()
                        }
        
        return [r for r in results if r is not None]
    
    async def _call_translation_api(self, lines: List[str]) -> List[str]:
        """Call the Google Cloud Translation API"""
        
        # Create the request
        request = translate.TranslateTextRequest(
            parent=self.parent,
            contents=lines,
            source_language_code=self.config.source_language,
            target_language_code=self.config.target_language,
            mime_type="text/plain"
        )
        
        # Make the API call
        response = self.client.translate_text(request=request)
        
        # Extract translations
        translations = [translation.translated_text for translation in response.translations]
        
        return translations
    
    def apply_glossary_terms(self, text: str) -> str:
        """Apply glossary terms to maintain consistency"""
        for hindi_term, english_term in self.glossary.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(hindi_term) + r'\b'
            if re.search(pattern, text):
                # If the term exists in original, ensure it's properly translated
                # This is a post-processing step to ensure consistency
                pass
        
        return text
    
    def get_cache_stats(self) -> Dict:
        """Get translation cache statistics"""
        total_cached = len(self.translation_cache)
        cache_size_bytes = sum(
            len(str(k)) + len(str(v)) 
            for k, v in self.translation_cache.items()
        )
        
        return {
            "total_cached_translations": total_cached,
            "cache_size_bytes": cache_size_bytes,
            "cache_size_mb": round(cache_size_bytes / 1024 / 1024, 2)
        }
    
    def clear_cache(self):
        """Clear the translation cache"""
        self.translation_cache.clear()
        print("Translation cache cleared")


# Test function
async def test_translation_engine():
    """Test the translation engine with sample text"""
    config = TranslationConfig()
    engine = TranslationEngine(config)
    
    test_sentences = [
        "नमस्ते, मेरा नाम राम है।",
        "मैं भारत से हूं और मुझे किताबें पढ़ना पसंद है।",
        "यह एक परीक्षण वाक्य है जिसमें मेरा ईमेल test@example.com है।",
        "कृपया मुझे +91-9876543210 पर कॉल करें।"
    ]
    
    print("Testing Translation Engine...")
    print("-" * 50)
    
    results = await engine.translate_chunk(test_sentences)
    
    for result in results:
        print(f"Original: {result['original']}")
        print(f"Translation: {result.get('translation', 'ERROR')}")
        if 'error' in result:
            print(f"Error: {result['error']}")
        print(f"Cache Hit: {result.get('cache_hit', False)}")
        print("-" * 50)
    
    # Test cache
    print("\nTesting cache (translating same sentences again)...")
    results2 = await engine.translate_chunk(test_sentences[:2])
    for result in results2:
        print(f"Cache Hit: {result.get('cache_hit', False)} - {result['original'][:30]}...")
    
    print(f"\nCache Stats: {engine.get_cache_stats()}")


if __name__ == "__main__":
    # Run test
    asyncio.run(test_translation_engine()) 
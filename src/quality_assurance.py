"""
Quality Assurance System for Hindi-English Book Translation
Implements comprehensive QA checks including MQM defect scoring and back-translation
"""

from typing import List, Dict, Tuple, Optional
import re
from difflib import SequenceMatcher
import spacy
from collections import Counter
import numpy as np
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
import logging
from datetime import datetime
import json
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class QAConfig:
    """Configuration for quality assurance parameters"""
    min_length_ratio: float = 0.8
    max_length_ratio: float = 1.5
    min_glossary_hit_rate: float = 0.98
    min_cosine_similarity: float = 0.85
    max_mqm_defects_per_1000: int = 3
    max_critical_errors: int = 0
    min_proper_noun_consistency: float = 0.95
    min_number_consistency: float = 1.0
    punctuation_threshold: float = 0.9


class QualityAssurance:
    """Quality checks for translations with MQM-based scoring"""
    
    def __init__(self, config: QAConfig = None):
        """Initialize QA system with language models"""
        self.config = config or QAConfig()
        
        # Load spaCy models for both languages
        try:
            self.nlp_en = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("English spaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp_en = None
            
        try:
            self.nlp_hi = spacy.load("xx_sent_ud_sm")  # Multi-language sentence detection
        except OSError:
            logger.warning("Multi-language spaCy model not found. Run: python -m spacy download xx_sent_ud_sm")
            self.nlp_hi = None
            
        # Load NLLB model for back-translation similarity
        try:
            self.nllb_model = SentenceTransformer('facebook/nllb-200-distilled-600M')
        except Exception as e:
            logger.warning(f"Failed to load NLLB model: {e}")
            self.nllb_model = None
        
        # Translation client for back-translation (will be set by controller)
        self.translation_client = None
        self.translation_endpoint = None
        
        # Define quality check methods
        self.checks = [
            self.check_length_ratio,
            self.check_number_consistency,
            self.check_proper_nouns,
            self.check_punctuation,
            self.check_terminology_consistency,
            self.check_sentence_structure,
            self.check_back_translation_similarity,
            self.check_glossary_hit_rate
        ]
        
        # MQM defect categories
        self.mqm_categories = {
            'accuracy': ['mistranslation', 'omission', 'untranslated'],
            'fluency': ['grammar', 'spelling', 'typography', 'unintelligible'],
            'terminology': ['inconsistent', 'incorrect'],
            'style': ['register', 'awkward'],
            'locale': ['number_format', 'date_format']
        }
    
    async def run_quality_checks(self, source: str, translation: str,
                                glossary: Dict = None, 
                                translation_engine=None) -> Dict:
        """Run all quality checks on translation"""
        
        # Set translation engine if provided
        if translation_engine:
            self.translation_client = translation_engine.client
            self.translation_endpoint = translation_engine.endpoint
        
        results = {
            "overall_score": 1.0,
            "issues": [],
            "warnings": [],
            "suggestions": [],
            "mqm_defects": 0,
            "critical_errors": 0,
            "defect_details": [],
            "check_results": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Run each check
        for check in self.checks:
            try:
                if asyncio.iscoroutinefunction(check):
                    check_result = await check(source, translation, glossary)
                else:
                    check_result = check(source, translation, glossary)
                
                # Store individual check results
                check_name = check.__name__
                results["check_results"][check_name] = check_result
                
                # Aggregate results
                results["overall_score"] *= check_result.get("score", 1.0)
                
                if check_result.get("issues"):
                    results["issues"].extend(check_result["issues"])
                    results["mqm_defects"] += len(check_result["issues"])
                    
                    # Add to defect details
                    for issue in check_result["issues"]:
                        results["defect_details"].append({
                            "check": check_name,
                            "issue": issue,
                            "severity": check_result.get("severity", "minor")
                        })
                
                if check_result.get("warnings"):
                    results["warnings"].extend(check_result["warnings"])
                
                if check_result.get("suggestions"):
                    results["suggestions"].extend(check_result["suggestions"])
                
                if check_result.get("critical"):
                    results["critical_errors"] += 1
                    
            except Exception as e:
                logger.error(f"Error in check {check.__name__}: {str(e)}")
                results["warnings"].append(f"Check {check.__name__} failed: {str(e)}")
        
        # Calculate MQM score
        results["mqm_score"] = self._calculate_mqm_score(results["defect_details"], 
                                                        len(source.split()))
        
        # Final verdict
        results["passed"] = (
            results["critical_errors"] == 0 and
            results["mqm_defects"] <= self.config.max_mqm_defects_per_1000 * (len(source.split()) / 1000)
        )
        
        return results
    
    def check_length_ratio(self, source: str, translation: str, 
                          glossary: Dict = None) -> Dict:
        """Check if translation length is within acceptable range"""
        
        source_len = len(source.split())
        trans_len = len(translation.split())
        
        if source_len == 0:
            return {"score": 0.0, "issues": ["Source text is empty"], "critical": True}
        
        ratio = trans_len / source_len
        
        result = {"score": 1.0, "issues": [], "warnings": []}
        
        if ratio < self.config.min_length_ratio:
            result["score"] = ratio / self.config.min_length_ratio
            result["issues"].append(
                f"Translation too short: {ratio:.2f} ratio (expected >= {self.config.min_length_ratio})"
            )
        elif ratio > self.config.max_length_ratio:
            result["score"] = self.config.max_length_ratio / ratio
            result["issues"].append(
                f"Translation too long: {ratio:.2f} ratio (expected <= {self.config.max_length_ratio})"
            )
        
        result["ratio"] = ratio
        return result
    
    def check_number_consistency(self, source: str, translation: str,
                                glossary: Dict = None) -> Dict:
        """Check if numbers are preserved correctly"""
        
        # Extract numbers from both texts
        source_numbers = re.findall(r'\d+(?:\.\d+)?', source)
        trans_numbers = re.findall(r'\d+(?:\.\d+)?', translation)
        
        result = {"score": 1.0, "issues": [], "warnings": []}
        
        # Check if all source numbers appear in translation
        source_num_set = set(source_numbers)
        trans_num_set = set(trans_numbers)
        
        missing_numbers = source_num_set - trans_num_set
        extra_numbers = trans_num_set - source_num_set
        
        if missing_numbers:
            result["score"] = 0.5
            result["issues"].append(
                f"Missing numbers in translation: {', '.join(missing_numbers)}"
            )
            result["severity"] = "major"
        
        if extra_numbers:
            result["warnings"].append(
                f"Extra numbers in translation: {', '.join(extra_numbers)}"
            )
        
        return result
    
    def check_proper_nouns(self, source: str, translation: str,
                          glossary: Dict = None) -> Dict:
        """Check proper noun consistency"""
        
        result = {"score": 1.0, "issues": [], "warnings": []}
        
        if not self.nlp_en or not self.nlp_hi:
            result["warnings"].append("spaCy models not available for proper noun check")
            return result
        
        # Extract proper nouns from source (Hindi)
        # For Hindi, we'll use capitalization and context clues
        source_proper_nouns = self._extract_hindi_proper_nouns(source)
        
        # Extract proper nouns from translation
        doc_en = self.nlp_en(translation)
        trans_proper_nouns = [ent.text for ent in doc_en.ents if ent.label_ in ["PERSON", "GPE", "ORG"]]
        
        # Check consistency
        if source_proper_nouns:
            found_count = sum(1 for noun in source_proper_nouns 
                            if any(noun.lower() in trans.lower() for trans in trans_proper_nouns))
            consistency_ratio = found_count / len(source_proper_nouns)
            
            if consistency_ratio < self.config.min_proper_noun_consistency:
                result["score"] = consistency_ratio
                result["issues"].append(
                    f"Proper noun consistency: {consistency_ratio:.2%} (expected >= {self.config.min_proper_noun_consistency:.2%})"
                )
        
        return result
    
    def check_punctuation(self, source: str, translation: str,
                         glossary: Dict = None) -> Dict:
        """Check punctuation preservation"""
        
        # Count punctuation marks
        source_punct = Counter(re.findall(r'[।\.\?!,;:\-\(\)\[\]\"\'""'']', source))
        trans_punct = Counter(re.findall(r'[\.\?!,;:\-\(\)\[\]\"\'""'']', translation))
        
        result = {"score": 1.0, "issues": [], "warnings": []}
        
        # Check sentence endings (। in Hindi should map to . in English)
        hindi_sentence_endings = source_punct.get('।', 0)
        english_sentence_endings = trans_punct.get('.', 0)
        
        if hindi_sentence_endings > 0:
            ending_ratio = english_sentence_endings / hindi_sentence_endings
            if ending_ratio < 0.8 or ending_ratio > 1.2:
                result["score"] *= 0.8
                result["issues"].append(
                    f"Sentence ending mismatch: {hindi_sentence_endings} Hindi (।) vs {english_sentence_endings} English (.)"
                )
        
        # Check question marks
        source_questions = source_punct.get('?', 0)
        trans_questions = trans_punct.get('?', 0)
        
        if source_questions != trans_questions:
            result["score"] *= 0.9
            result["warnings"].append(
                f"Question mark mismatch: {source_questions} source vs {trans_questions} translation"
            )
        
        return result
    
    def check_terminology_consistency(self, source: str, translation: str,
                                    glossary: Dict = None) -> Dict:
        """Check terminology consistency across translation"""
        
        result = {"score": 1.0, "issues": [], "warnings": []}
        
        # Extract repeated terms (words appearing 2+ times)
        source_words = source.split()
        source_freq = Counter(word for word in source_words if len(word) > 3)
        repeated_terms = {word: count for word, count in source_freq.items() if count >= 2}
        
        if repeated_terms and translation:
            # Check if repeated terms are translated consistently
            # This is a simplified check - in production, would use alignment models
            trans_words = translation.split()
            trans_freq = Counter(word for word in trans_words if len(word) > 3)
            
            # If a term appears N times in source, its translation should appear ~N times too
            consistency_issues = []
            for term, count in repeated_terms.items():
                # Find potential translations (this is simplified)
                max_trans_count = max(trans_freq.values()) if trans_freq else 0
                if max_trans_count < count * 0.7:  # Allow some variation
                    consistency_issues.append(term)
            
            if consistency_issues:
                result["score"] = 0.8
                result["warnings"].append(
                    f"Potential terminology inconsistency for: {', '.join(consistency_issues[:3])}"
                )
        
        return result
    
    def check_sentence_structure(self, source: str, translation: str,
                               glossary: Dict = None) -> Dict:
        """Check if sentence structure is preserved"""
        
        result = {"score": 1.0, "issues": [], "warnings": []}
        
        # Count sentences
        source_sentences = len(re.split(r'[।\.\?!]+', source))
        trans_sentences = len(re.split(r'[\.\?!]+', translation))
        
        if source_sentences > 0:
            sentence_ratio = trans_sentences / source_sentences
            
            if sentence_ratio < 0.8 or sentence_ratio > 1.2:
                result["score"] = 0.8
                result["issues"].append(
                    f"Sentence count mismatch: {source_sentences} source vs {trans_sentences} translation"
                )
                result["severity"] = "minor"
        
        return result
    
    async def check_back_translation_similarity(self, source: str, translation: str,
                                              glossary: Dict = None) -> Dict:
        """Check semantic similarity using back-translation"""
        
        result = {"score": 1.0, "issues": [], "warnings": []}
        
        if not self.nllb_model:
            result["warnings"].append("NLLB model not available for back-translation check")
            return result
        
        if not self.translation_client:
            result["warnings"].append("Translation client not available for back-translation")
            return result
        
        try:
            # Back-translate English to Hindi
            back_translation_instances = [{
                "model": "projects/${PROJECT_ID}/locations/us-central1/models/general/translation-llm",
                "source_language_code": "en",
                "target_language_code": "hi",
                "contents": [translation],
                "mimeType": "text/plain"
            }]
            
            back_trans_response = await self.translation_client.predict(
                instances=back_translation_instances,
                endpoint=self.translation_endpoint
            )
            
            back_translated_hindi = back_trans_response.predictions[0]["translations"][0]["translatedText"]
            
            # Compare embeddings
            source_emb = self.nllb_model.encode(source, convert_to_tensor=True)
            back_trans_emb = self.nllb_model.encode(back_translated_hindi, convert_to_tensor=True)
            
            # Calculate cosine similarity
            cosine_sim = float(np.dot(source_emb, back_trans_emb) / (
                np.linalg.norm(source_emb) * np.linalg.norm(back_trans_emb)
            ))
            
            result["cosine_similarity"] = cosine_sim
            
            if cosine_sim < self.config.min_cosine_similarity:
                result["score"] = cosine_sim / self.config.min_cosine_similarity
                result["warnings"].append(
                    f"Back-translation similarity low: {cosine_sim:.2f} (expected >= {self.config.min_cosine_similarity})"
                )
                
                # Only mark as issue if very low
                if cosine_sim < 0.7:
                    result["issues"].append(
                        f"Potential semantic drift detected (similarity: {cosine_sim:.2f})"
                    )
                    result["severity"] = "major"
            
        except Exception as e:
            logger.error(f"Back-translation check failed: {str(e)}")
            result["warnings"].append(f"Back-translation check failed: {str(e)}")
            result["score"] = 0.9  # Don't penalize too much for technical failure
        
        return result
    
    def check_glossary_hit_rate(self, source: str, translation: str,
                               glossary: Dict = None) -> Dict:
        """Ensure glossary terms are used correctly"""
        
        result = {"score": 1.0, "issues": [], "warnings": []}
        
        if not glossary:
            return result
        
        hits = 0
        expected = 0
        missing_terms = []
        
        for term, expected_translation in glossary.items():
            if term in source:
                expected += 1
                # Check if any variant of expected translation appears
                variants = expected_translation.split('/')
                found = False
                for variant in variants:
                    variant = variant.strip()
                    if variant.lower() in translation.lower():
                        hits += 1
                        found = True
                        break
                
                if not found:
                    missing_terms.append(f"{term} → {expected_translation}")
        
        if expected > 0:
            hit_rate = hits / expected
            result["hit_rate"] = hit_rate
            
            if hit_rate < self.config.min_glossary_hit_rate:
                result["score"] = hit_rate / self.config.min_glossary_hit_rate
                result["issues"].append(
                    f"Glossary hit rate: {hit_rate:.2%} (expected >= {self.config.min_glossary_hit_rate:.2%})"
                )
                result["severity"] = "major"
                
                if missing_terms:
                    result["issues"].append(
                        f"Missing glossary terms: {', '.join(missing_terms[:3])}"
                        + (" ..." if len(missing_terms) > 3 else "")
                    )
        
        return result
    
    def _extract_hindi_proper_nouns(self, text: str) -> List[str]:
        """Extract likely proper nouns from Hindi text"""
        
        proper_nouns = []
        
        # Common Hindi name patterns
        name_patterns = [
            r'श्री\s+(\w+)',  # Shri + name
            r'(\w+)\s+जी',     # name + ji
            r'डॉ\.\s+(\w+)',   # Dr. + name
            r'प्रो\.\s+(\w+)', # Prof. + name
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, text)
            proper_nouns.extend(matches)
        
        # Also look for capitalized words in Devanagari
        # (This is simplified - proper implementation would use NER)
        words = text.split()
        for word in words:
            if len(word) > 2 and word[0].isupper():
                proper_nouns.append(word)
        
        return list(set(proper_nouns))
    
    def _calculate_mqm_score(self, defect_details: List[Dict], word_count: int) -> float:
        """Calculate MQM score based on defects per 1000 words"""
        
        if word_count == 0:
            return 0.0
        
        # Weight defects by severity
        severity_weights = {
            'critical': 5,
            'major': 3,
            'minor': 1
        }
        
        weighted_defects = sum(
            severity_weights.get(defect.get('severity', 'minor'), 1)
            for defect in defect_details
        )
        
        # Calculate defects per 1000 words
        defects_per_1000 = (weighted_defects / word_count) * 1000
        
        # Convert to score (0-1 scale)
        # Perfect = 1.0, max allowed = 0.7, beyond = proportionally lower
        if defects_per_1000 <= self.config.max_mqm_defects_per_1000:
            score = 1.0 - (defects_per_1000 / (self.config.max_mqm_defects_per_1000 * 2))
        else:
            score = 0.5 * (self.config.max_mqm_defects_per_1000 / defects_per_1000)
        
        return max(0.0, min(1.0, score))
    
    def generate_qa_report(self, qa_results: List[Dict]) -> Dict:
        """Generate summary QA report for a batch of translations"""
        
        total_checks = len(qa_results)
        if total_checks == 0:
            return {"error": "No QA results to analyze"}
        
        # Aggregate statistics
        total_defects = sum(r.get("mqm_defects", 0) for r in qa_results)
        total_critical = sum(r.get("critical_errors", 0) for r in qa_results)
        avg_score = sum(r.get("overall_score", 0) for r in qa_results) / total_checks
        
        # Count check failures
        check_failures = Counter()
        for result in qa_results:
            for check_name, check_result in result.get("check_results", {}).items():
                if check_result.get("issues"):
                    check_failures[check_name] += 1
        
        # Find common issues
        all_issues = []
        for result in qa_results:
            all_issues.extend(result.get("issues", []))
        
        issue_counts = Counter(all_issues)
        
        report = {
            "summary": {
                "total_segments": total_checks,
                "average_score": avg_score,
                "total_defects": total_defects,
                "critical_errors": total_critical,
                "passed": total_critical == 0 and avg_score >= 0.85
            },
            "check_statistics": {
                check: {
                    "failures": count,
                    "failure_rate": count / total_checks
                }
                for check, count in check_failures.most_common()
            },
            "common_issues": issue_counts.most_common(10),
            "recommendations": self._generate_recommendations(check_failures, avg_score),
            "timestamp": datetime.now().isoformat()
        }
        
        return report
    
    def _generate_recommendations(self, check_failures: Counter, avg_score: float) -> List[str]:
        """Generate recommendations based on QA results"""
        
        recommendations = []
        
        if avg_score < 0.7:
            recommendations.append("Consider re-translation with adjusted parameters")
        
        if check_failures.get("check_glossary_hit_rate", 0) > 0:
            recommendations.append("Review and enforce glossary terms in translation")
        
        if check_failures.get("check_back_translation_similarity", 0) > 0:
            recommendations.append("Check for semantic drift in translations")
        
        if check_failures.get("check_length_ratio", 0) > 0:
            recommendations.append("Adjust translation verbosity settings")
        
        if check_failures.get("check_number_consistency", 0) > 0:
            recommendations.append("Ensure numeric values are preserved accurately")
        
        return recommendations 
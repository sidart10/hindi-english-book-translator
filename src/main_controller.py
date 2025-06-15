#!/usr/bin/env python3
"""
Main Controller for Hindi Book Translation System
Orchestrates the translation process with batch processing and concurrency control
"""

import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import aiohttp
from asyncio import Semaphore
# Import our components
from .translation_engine import TranslationEngine, TranslationConfig
from .document_processor import DocumentProcessor
from .cost_meter import CostMeter
from .quality_assurance import QualityAssurance, QAConfig
from .latex_output import LaTeXOutputGenerator


class BookTranslationController:
    """Main controller for book translation workflow"""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize controller with configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self.load_config(config_path)
        
        # Initialize components
        self.translation_engine = TranslationEngine(
            TranslationConfig(**self.config.get("translation", {}))
        )
        self.document_processor = DocumentProcessor()
        self.cost_meter = CostMeter(
            monthly_budget=self.config.get("cost", {}).get("monthly_budget", 1000),
            alert_threshold=self.config.get("cost", {}).get("alert_threshold", 0.7)
        )
        
        # Initialize Quality Assurance
        qa_config = self.config.get("quality", {})
        self.quality_assurance = QualityAssurance(
            QAConfig(
                min_length_ratio=qa_config.get("min_length_ratio", 0.8),
                max_length_ratio=qa_config.get("max_length_ratio", 1.5),
                min_glossary_hit_rate=qa_config.get("min_glossary_hit_rate", 0.98),
                min_cosine_similarity=qa_config.get("min_cosine_similarity", 0.85),
                max_mqm_defects_per_1000=qa_config.get("max_mqm_defects_per_1000", 3),
                max_critical_errors=qa_config.get("max_critical_errors", 0)
            )
        )
        
        # Set up concurrency control
        self.semaphore = Semaphore(
            self.config.get("processing", {}).get("quota_limit", 10)
        )
        
        # Processing parameters
        self.batch_size = self.config.get("processing", {}).get("batch_size", 50)
        self.max_retries = self.config.get("processing", {}).get("max_retries", 3)
        
        # Initialize LaTeX generator
        self.latex_generator = LaTeXOutputGenerator(
            title=self.config.get("output", {}).get("title", "Hindi to English Translation"),
            author=self.config.get("output", {}).get("author", "Translation System"),
            document_class=self.config.get("output", {}).get("document_class", "book")
        )
        
        print(f"Controller initialized with batch_size={self.batch_size}, "
              f"concurrency={self.config.get('processing', {}).get('quota_limit', 10)}")
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from file"""
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                "translation": {
                    "source_language": "hi",
                    "target_language": "en",
                    "temperature": 0,
                    "max_characters": 9000
                },
                "processing": {
                    "batch_size": 50,
                    "quota_limit": 10,
                    "max_retries": 3
                },
                "cost": {
                    "monthly_budget": 1000,
                    "alert_threshold": 0.7
                },
                "quality": {
                    "min_confidence": 0.85,
                    "min_length_ratio": 0.8,
                    "max_length_ratio": 1.5,
                    "min_glossary_hit_rate": 0.98,
                    "min_cosine_similarity": 0.85,
                    "max_mqm_defects_per_1000": 3,
                    "max_critical_errors": 0
                }
            }
    
    async def translate_book(self, input_path: str, output_path: str,
                           project_id: Optional[str] = None) -> Dict:
        """
        Main method to translate entire book
        
        Args:
            input_path: Path to input document
            output_path: Path for output LaTeX file
            project_id: Optional project identifier
            
        Returns:
            Dictionary with translation results and statistics
        """
        if not project_id:
            project_id = f"book_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n📚 Starting translation of '{input_path}'")
        print(f"   Project ID: {project_id}")
        
        results = {
            "project_id": project_id,
            "input_file": input_path,
            "output_file": output_path,
            "start_time": datetime.now().isoformat(),
            "statistics": {
                "total_sentences": 0,
                "translated_sentences": 0,
                "total_characters": 0,
                "total_cost": 0.0,
                "errors": [],
                "quality_warnings": [],
                "mqm_defects": 0,
                "critical_errors": 0,
                "average_quality_score": 0.0,
                "glossary_hit_rate": 0.0,
                "qa_passed": 0,
                "qa_failed": 0
            }
        }
        
        try:
            # Step 1: Process document into sentences
            print("\n📖 Extracting text from document...")
            sentences = list(self.document_processor.process_document(input_path))
            total_sentences = len(sentences)
            results["statistics"]["total_sentences"] = total_sentences
            
            if total_sentences == 0:
                raise ValueError("No sentences extracted from document")
            
            print(f"   Extracted {total_sentences} sentences")
            
            # Step 2: Generate segmentation JSONL
            segmentation_file = self._generate_segmentation_jsonl(sentences, project_id)
            print(f"   Generated segmentation file: {segmentation_file}")
            
            # Step 3: Translate in batches
            print(f"\n🔄 Translating in batches of {self.batch_size}...")
            translated_sentences = await self._translate_batches(sentences)
            
            results["statistics"]["translated_sentences"] = len(translated_sentences)
            results["statistics"]["total_cost"] = self.cost_meter.current_cost
            
            # Calculate QA statistics
            qa_results = []
            total_defects = 0
            total_critical = 0
            total_quality_score = 0.0
            qa_passed = 0
            qa_failed = 0
            
            for sent in translated_sentences:
                if "qa_result" in sent:
                    qa_result = sent["qa_result"]
                    qa_results.append(qa_result)
                    
                    total_defects += qa_result.get("mqm_defects", 0)
                    total_critical += qa_result.get("critical_errors", 0)
                    total_quality_score += qa_result.get("overall_score", 0)
                    
                    if qa_result.get("passed", False):
                        qa_passed += 1
                    else:
                        qa_failed += 1
            
            # Update statistics
            if qa_results:
                results["statistics"]["mqm_defects"] = total_defects
                results["statistics"]["critical_errors"] = total_critical
                results["statistics"]["average_quality_score"] = total_quality_score / len(qa_results)
                results["statistics"]["qa_passed"] = qa_passed
                results["statistics"]["qa_failed"] = qa_failed
                
                # Generate QA report
                qa_report = self.quality_assurance.generate_qa_report(qa_results)
                results["qa_report"] = qa_report
                
                print(f"\n📊 Quality Assurance Summary:")
                print(f"   Average Score: {results['statistics']['average_quality_score']:.2%}")
                print(f"   MQM Defects: {total_defects} ({total_defects / (len(sentences) / 1000):.1f} per 1000 sentences)")
                print(f"   Critical Errors: {total_critical}")
                print(f"   QA Passed: {qa_passed}/{len(qa_results)} ({qa_passed/len(qa_results)*100:.1f}%)")
            
            # Step 4: Generate output LaTeX
            print(f"\n📄 Generating LaTeX document...")
            self._generate_output(translated_sentences, output_path, qa_report)
            
            # Step 5: Final statistics
            results["end_time"] = datetime.now().isoformat()
            
            # Calculate total characters
            total_chars = 0
            for s in sentences:
                if hasattr(s, 'text'):
                    total_chars += len(s.text)
                else:
                    total_chars += len(s.get("text", ""))
            results["statistics"]["total_characters"] = total_chars
            
            print(f"\n✅ Translation complete!")
            print(f"   Total sentences: {total_sentences}")
            print(f"   Total characters: {total_chars:,}")
            print(f"   Total cost: ${self.cost_meter.current_cost:.2f}")
            print(f"   Output saved to: {output_path}")
            
            # Check for cost alerts
            status = self.cost_meter.get_status()
            if status["alerts"]:
                print("\n⚠️  Cost Alerts:")
                for alert in status["alerts"]:
                    print(f"   {alert}")
            
            return results
            
        except Exception as e:
            results["statistics"]["errors"].append(str(e))
            print(f"\n❌ Error during translation: {e}")
            raise
    
    def _generate_segmentation_jsonl(self, sentences: List[Dict], 
                                    project_id: str) -> str:
        """Generate JSONL file with segmented sentences"""
        jsonl_path = f"segmentation_{project_id}.jsonl"
        
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for sent in sentences:
                # Convert to dict if it's a DocumentSegment object
                if hasattr(sent, 'to_dict'):
                    sent_dict = sent.to_dict()
                else:
                    sent_dict = sent
                f.write(json.dumps(sent_dict, ensure_ascii=False) + '\n')
        
        return jsonl_path
    
    async def _translate_batches(self, sentences: List[Dict]) -> List[Dict]:
        """Translate sentences in batches with concurrency control"""
        translated_sentences = []
        total_batches = (len(sentences) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(0, len(sentences), self.batch_size):
            batch = sentences[batch_idx:batch_idx + self.batch_size]
            current_batch_num = (batch_idx // self.batch_size) + 1
            
            print(f"   Batch {current_batch_num}/{total_batches} "
                  f"({len(batch)} sentences)")
            
            # Check cost threshold before translating
            if self.cost_meter.check_threshold(0.7):
                cost_status = self.cost_meter.get_status()
                print(f"   ⚠️  {cost_status['alerts'][0]}")
            
            # Extract text from sentence objects
            batch_texts = []
            for s in batch:
                if hasattr(s, 'text'):
                    batch_texts.append(s.text)
                else:
                    batch_texts.append(s.get("text", ""))
            
            # Translate with rate limiting and retry logic
            translated = await self._translate_with_retry(batch_texts, batch_idx)
            
            # Combine original and translated data
            for sent_obj, trans_result in zip(batch, translated):
                # Convert DocumentSegment to dict if needed
                if hasattr(sent_obj, 'to_dict'):
                    sent_dict = sent_obj.to_dict()
                else:
                    sent_dict = sent_obj
                
                sent_dict["translation"] = trans_result.get("translation", "")
                sent_dict["confidence"] = trans_result.get("confidence", 0.0)
                
                # Run quality assurance checks
                source_text = sent_dict.get("text", "")
                translation_text = trans_result.get("translation", "")
                
                if source_text and translation_text:
                    qa_result = await self.quality_assurance.run_quality_checks(
                        source_text,
                        translation_text,
                        self.translation_engine.glossary,
                        self.translation_engine
                    )
                    
                    sent_dict["qa_result"] = qa_result
                    
                    # Add quality warnings
                    if not qa_result["passed"]:
                        sent_dict["quality_warning"] = f"QA Failed: Score {qa_result['overall_score']:.2f}"
                        if qa_result["issues"]:
                            sent_dict["quality_issues"] = qa_result["issues"]
                    elif qa_result["warnings"]:
                        sent_dict["quality_warning"] = f"QA Warnings: {len(qa_result['warnings'])} found"
                
                translated_sentences.append(sent_dict)
            
            # Update cost tracking
            total_chars = sum(len(t) for t in batch_texts)
            cost_result = self.cost_meter.add_cost(
                total_chars, 
                f"Batch {current_batch_num}"
            )
            
            print(f"      Cost: ${cost_result['cost']:.4f} "
                  f"(Total: ${cost_result['total_cost']:.2f})")
        
        return translated_sentences
    
    async def _translate_with_retry(self, texts: List[str], 
                                   batch_idx: int) -> List[Dict]:
        """Translate with retry logic for rate limiting and errors"""
        for retry in range(self.max_retries):
            try:
                async with self.semaphore:
                    # Call translation engine
                    results = await self.translation_engine.translate_chunk(texts)
                    return results
                    
            except aiohttp.ClientResponseError as e:
                if e.status == 429:  # Rate limit
                    wait_time = 2 ** retry * 10  # Exponential backoff
                    print(f"      Rate limit hit, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    
                elif e.status >= 500:  # Server error
                    wait_time = 2 ** retry * 5
                    print(f"      Server error, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    
                else:
                    raise  # Other errors, don't retry
                    
            except Exception as e:
                if retry == self.max_retries - 1:
                    print(f"      Failed after {self.max_retries} retries: {e}")
                    # Return empty translations for this batch
                    return [{"translation": "", "error": str(e)} for _ in texts]
                else:
                    wait_time = 2 ** retry * 2
                    print(f"      Error: {e}, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
        
        # Should not reach here
        return [{"translation": "", "error": "Max retries exceeded"} for _ in texts]
    
    def _generate_output(self, translated_sentences: List[Dict], 
                        output_path: str,
                        qa_report: Optional[Dict] = None):
        """Generate LaTeX output with translations"""
        # Update LaTeX generator title if available
        input_filename = Path(self.config.get("input_file", "Unknown")).stem
        self.latex_generator.title = self._escape_for_latex(input_filename)
        
        # Generate main LaTeX document
        latex_path = self.latex_generator.generate_latex(
            translated_sentences,
            output_path,
            include_source=self.config.get("output", {}).get("include_source", False),
            include_qa_details=self.config.get("output", {}).get("include_qa_details", True)
        )
        
        # If we have a QA report, append it
        if qa_report:
            appendix = self.latex_generator.generate_quality_report_appendix(qa_report)
            
            # Append to the LaTeX file (before \end{document})
            with open(latex_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Insert appendix before \end{document}
            content = content.replace("\\end{document}", appendix + "\n\\end{document}")
            
            with open(latex_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"   Quality report appended to LaTeX document")
    
    def _escape_for_latex(self, text: str) -> str:
        """Helper to escape text for LaTeX"""
        if not text:
            return ""
        
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
        
        for char, replacement in special_chars.items():
            text = text.replace(char, replacement)
        
        return text


# Test function
async def test_controller():
    """Test the main controller with a sample text file"""
    print("Testing Main Controller...")
    print("=" * 70)
    
    # Create a test text file
    test_file = "test_controller_input.txt"
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write("नमस्ते, मेरा नाम राम है।\n")
        f.write("मैं भारत से हूं।\n")
        f.write("यह एक परीक्षण है।\n")
        f.write("आज मौसम बहुत अच्छा है।\n")
        f.write("मुझे हिंदी भाषा पसंद है।\n")
    
    # Initialize controller
    controller = BookTranslationController()
    
    # Run translation
    output_file = "test_controller_output.tex"
    
    try:
        results = await controller.translate_book(
            test_file,
            output_file,
            project_id="test_001"
        )
        
        print("\n" + "=" * 70)
        print("Translation Results:")
        print(json.dumps(results, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up test files
        if Path(test_file).exists():
            os.remove(test_file)
        if Path(f"segmentation_test_001.jsonl").exists():
            os.remove(f"segmentation_test_001.jsonl")


if __name__ == "__main__":
    # Run test
    asyncio.run(test_controller()) 
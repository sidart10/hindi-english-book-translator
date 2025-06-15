#!/usr/bin/env python3
"""
Document Processor for Hindi Book Translation System
Handles PDF, EPUB, TXT, and DOCX files with sentence extraction
"""

import fitz  # PyMuPDF
import re
import json
from typing import List, Dict, Generator, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import pytesseract
from PIL import Image
import io
import docx
import ebooklib
from ebooklib import epub
from datetime import datetime


@dataclass
class DocumentSegment:
    """Represents a segment of text from a document"""
    page: int
    sentence: int
    text: str
    type: str = "body"  # body, header, footer, caption
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        if data['metadata'] is None:
            data['metadata'] = {}
        return data


class DocumentProcessor:
    """Handles various document formats and extracts text"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.epub', '.txt', '.docx']
        self.page_cache = {}
        self.min_sentence_length = 10  # Minimum characters for a valid sentence
        
        # Sentence delimiters for Hindi and English
        self.sentence_delimiters = {
            'hindi': r'[।॥\.\?!]+',  # Hindi uses । (purna viram) and ॥ (double danda)
            'english': r'[\.\?!]+',
            'mixed': r'[।॥\.\?!]+'  # Combined pattern for mixed text
        }
    
    def process_document(self, file_path: str, output_jsonl: Optional[str] = None) -> Generator[DocumentSegment, None, None]:
        """Process document and yield segments for translation"""
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported format: {file_extension}")
        
        # Create generator based on file type
        if file_extension == '.pdf':
            segments = self._process_pdf(str(file_path))
        elif file_extension == '.epub':
            segments = self._process_epub(str(file_path))
        elif file_extension == '.txt':
            segments = self._process_text(str(file_path))
        elif file_extension == '.docx':
            segments = self._process_docx(str(file_path))
        else:
            raise ValueError(f"Unsupported format: {file_extension}")
        
        # If output JSONL is specified, write segments
        if output_jsonl:
            with open(output_jsonl, 'w', encoding='utf-8') as f:
                for segment in segments:
                    f.write(json.dumps(segment.to_dict(), ensure_ascii=False) + '\n')
                    yield segment
        else:
            # Just yield segments
            yield from segments
    
    def _process_pdf(self, file_path: str) -> Generator[DocumentSegment, None, None]:
        """Extract text from PDF with OCR support for scanned pages"""
        
        doc = fitz.open(file_path)
        print(f"Processing PDF: {file_path}")
        print(f"Total pages: {len(doc)}")
        
        for page_num, page in enumerate(doc):
            # Try text extraction first
            text = page.get_text()
            
            # Check if page has minimal text (might be scanned)
            if len(text.strip()) < 50:
                print(f"Page {page_num + 1} appears to be scanned, attempting OCR...")
                # Try OCR
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                try:
                    # OCR with Hindi language support
                    text = pytesseract.image_to_string(img, lang='hin+eng')
                    ocr_used = True
                except Exception as e:
                    print(f"OCR failed for page {page_num + 1}: {str(e)}")
                    ocr_used = False
            else:
                ocr_used = False
            
            # Split into sentences
            sentences = self._split_into_sentences(text)
            
            for sent_idx, sentence in enumerate(sentences):
                if sentence.strip() and len(sentence.strip()) >= self.min_sentence_length:
                    segment = DocumentSegment(
                        page=page_num + 1,
                        sentence=sent_idx + 1,
                        text=sentence.strip(),
                        type="body",
                        metadata={
                            "source": "pdf",
                            "ocr_used": ocr_used,
                            "file": file_path
                        }
                    )
                    yield segment
        
        doc.close()
        print(f"PDF processing complete")
    
    def _process_text(self, file_path: str) -> Generator[DocumentSegment, None, None]:
        """Process plain text file"""
        
        print(f"Processing text file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into paragraphs first
        paragraphs = content.split('\n\n')
        
        page_num = 1
        global_sentence_num = 0
        
        for para in paragraphs:
            if para.strip():
                sentences = self._split_into_sentences(para)
                
                for sentence in sentences:
                    if sentence.strip() and len(sentence.strip()) >= self.min_sentence_length:
                        global_sentence_num += 1
                        segment = DocumentSegment(
                            page=page_num,
                            sentence=global_sentence_num,
                            text=sentence.strip(),
                            type="body",
                            metadata={
                                "source": "txt",
                                "file": file_path
                            }
                        )
                        yield segment
    
    def _process_docx(self, file_path: str) -> Generator[DocumentSegment, None, None]:
        """Process Microsoft Word document"""
        
        print(f"Processing DOCX file: {file_path}")
        
        doc = docx.Document(file_path)
        
        page_num = 1
        global_sentence_num = 0
        
        for para in doc.paragraphs:
            if para.text.strip():
                sentences = self._split_into_sentences(para.text)
                
                for sentence in sentences:
                    if sentence.strip() and len(sentence.strip()) >= self.min_sentence_length:
                        global_sentence_num += 1
                        
                        # Determine type based on style
                        para_type = "body"
                        if para.style.name.startswith('Heading'):
                            para_type = "header"
                        
                        segment = DocumentSegment(
                            page=page_num,
                            sentence=global_sentence_num,
                            text=sentence.strip(),
                            type=para_type,
                            metadata={
                                "source": "docx",
                                "style": para.style.name,
                                "file": file_path
                            }
                        )
                        yield segment
    
    def _process_epub(self, file_path: str) -> Generator[DocumentSegment, None, None]:
        """Process EPUB file"""
        
        print(f"Processing EPUB file: {file_path}")
        
        book = epub.read_epub(file_path)
        
        page_num = 1
        global_sentence_num = 0
        
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                content = item.get_content().decode('utf-8', errors='ignore')
                
                # Remove HTML tags (basic approach)
                text = re.sub('<[^<]+?>', '', content)
                
                if text.strip():
                    sentences = self._split_into_sentences(text)
                    
                    for sentence in sentences:
                        if sentence.strip() and len(sentence.strip()) >= self.min_sentence_length:
                            global_sentence_num += 1
                            segment = DocumentSegment(
                                page=page_num,
                                sentence=global_sentence_num,
                                text=sentence.strip(),
                                type="body",
                                metadata={
                                    "source": "epub",
                                    "chapter": item.get_name(),
                                    "file": file_path
                                }
                            )
                            yield segment
                
                page_num += 1
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for Hindi/English mixed content"""
        
        # Clean up text
        text = text.strip()
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Replace em-dashes with en-dashes (per style guide)
        text = text.replace('—', '–')
        
        # Detect primary language
        hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
        total_chars = len(text)
        hindi_ratio = hindi_chars / total_chars if total_chars > 0 else 0
        
        # Choose delimiter pattern based on content
        if hindi_ratio > 0.5:
            delimiter_pattern = self.sentence_delimiters['hindi']
        else:
            delimiter_pattern = self.sentence_delimiters['mixed']
        
        # Split on sentence boundaries
        # Keep the delimiters with the sentences
        parts = re.split(f'({delimiter_pattern})', text)
        
        sentences = []
        current_sentence = ""
        
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Text part
                current_sentence = part.strip()
            else:  # Delimiter part
                if current_sentence:
                    # Add delimiter to sentence
                    current_sentence += part
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
        
        # Add any remaining sentence
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # Filter out very short segments that aren't real sentences
        filtered_sentences = []
        for sent in sentences:
            # Remove sentences that are just page numbers or headers
            if (len(sent) >= self.min_sentence_length and 
                not re.match(r'^[\d\s\-–—]*$', sent) and  # Not just numbers/dashes
                not re.match(r'^(Chapter|अध्याय|भाग)\s*[\d\s]*$', sent, re.I)):  # Not chapter headers
                filtered_sentences.append(sent)
        
        return filtered_sentences
    
    def generate_segmentation_jsonl(self, file_path: str, output_path: str) -> str:
        """Generate JSONL file with document segmentation"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not output_path:
            output_path = f"segmentation_{Path(file_path).stem}_{timestamp}.jsonl"
        
        segment_count = 0
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for segment in self.process_document(file_path):
                f.write(json.dumps(segment.to_dict(), ensure_ascii=False) + '\n')
                segment_count += 1
        
        print(f"Generated segmentation file: {output_path}")
        print(f"Total segments: {segment_count}")
        
        return output_path
    
    def load_segmentation_jsonl(self, jsonl_path: str) -> List[DocumentSegment]:
        """Load segments from JSONL file"""
        
        segments = []
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                segment = DocumentSegment(**data)
                segments.append(segment)
        
        return segments


# Test functions
def test_pdf_processing():
    """Test PDF processing with a sample file"""
    processor = DocumentProcessor()
    
    # Create a test PDF if needed
    test_pdf = "test_hindi.pdf"
    
    if not Path(test_pdf).exists():
        print("Creating test PDF...")
        # Create a simple test PDF
        doc = fitz.open()
        page = doc.new_page()
        
        # Add Hindi text
        text = """नमस्ते। यह एक परीक्षण दस्तावेज़ है।
        
इसमें कुछ हिंदी वाक्य हैं। हम देखेंगे कि पीडीएफ प्रोसेसिंग कैसे काम करती है।

यह तीसरा पैराग्राफ है। इसमें अधिक जानकारी है।"""
        
        page.insert_text((50, 50), text, fontname="Helvetica", fontsize=12)
        doc.save(test_pdf)
        doc.close()
        print(f"Created test PDF: {test_pdf}")
    
    # Process the PDF
    print("\nProcessing PDF...")
    segments = list(processor.process_document(test_pdf))
    
    print(f"\nExtracted {len(segments)} segments:")
    for i, segment in enumerate(segments[:5]):  # Show first 5
        print(f"\nSegment {i+1}:")
        print(f"  Page: {segment.page}")
        print(f"  Text: {segment.text}")
    
    # Generate JSONL
    jsonl_path = processor.generate_segmentation_jsonl(test_pdf, "test_segmentation.jsonl")
    
    return segments


def test_text_processing():
    """Test text file processing"""
    processor = DocumentProcessor()
    
    # Create test text file
    test_file = "test_hindi.txt"
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write("""यह एक परीक्षण फ़ाइल है। इसमें कई वाक्य हैं।

दूसरा पैराग्राफ यहाँ शुरू होता है। इसमें भी कुछ वाक्य हैं। क्या आप इसे पढ़ सकते हैं?

तीसरे पैराग्राफ में और जानकारी है। यह बहुत महत्वपूर्ण है।""")
    
    # Process
    segments = list(processor.process_document(test_file))
    
    print(f"\nExtracted {len(segments)} segments from text file:")
    for segment in segments:
        print(f"- {segment.text}")
    
    # Clean up
    Path(test_file).unlink()
    
    return segments


if __name__ == "__main__":
    print("Testing Document Processor...")
    print("=" * 50)
    
    # Test text processing
    print("\n1. Testing Text File Processing:")
    test_text_processing()
    
    # Test PDF processing
    print("\n2. Testing PDF Processing:")
    test_pdf_processing() 
# Hindi-English Book Translation System 📚

A powerful system for translating Hindi books to English using Google Cloud Translation API, with support for multiple file formats, cost tracking, and quality assurance.

## Features ✨

- **Multi-format Support**: PDF, EPUB, TXT, and DOCX files
- **Google Cloud Translation**: Leverages Google's Translation API for high-quality translations
- **Cost Tracking**: Real-time cost monitoring with budget alerts
- **Quality Assurance**: MQM-based quality metrics and glossary consistency checks
- **PII Protection**: Automatic scrubbing of phone numbers, emails, and Aadhaar numbers
- **OCR Support**: Handles scanned PDFs using pytesseract
- **Translation Memory**: SQLite-based caching to avoid retranslating content
- **Cultural Preservation**: Maintains honorifics and cultural terms with explanations
- **Future-Ready**: Pre-integrated support for Mistral's upcoming OCR API

## Project Structure 📁

```
book-translator/
├── src/
│   ├── translation_engine.py    # Core translation engine
│   ├── document_processor.py    # Document parsing and extraction
│   ├── cost_meter.py           # Cost tracking and budget management
│   ├── main_controller.py      # Main application controller
│   ├── quality_assurance.py    # QA and validation system
│   ├── translation_memory.py   # Translation caching
│   ├── mistral_ocr_processor.py # Future Mistral OCR integration
│   └── cli.py                  # Command-line interface
├── documentation/
│   ├── setup-instructions.md   # Detailed setup guide
│   ├── progress-summary.md     # Development progress tracking
│   └── mistral-ocr-integration.md # Mistral OCR analysis
├── tests/                      # Test files
├── setup/                      # Setup scripts and configs
└── requirements.txt           # Python dependencies
```

## Quick Start 🚀

### Prerequisites

- Python 3.8+
- Google Cloud account with billing enabled
- Google Cloud CLI (`gcloud`)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/book-translator.git
   cd book-translator
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure Google Cloud**
   ```bash
   # Authenticate with Google Cloud
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   
   # Create service account
   gcloud iam service-accounts create book-translator \
       --display-name="Book Translator Service"
   
   # Download credentials
   gcloud iam service-accounts keys create service-account.json \
       --iam-account=book-translator@YOUR_PROJECT_ID.iam.gserviceaccount.com
   ```

4. **Set up environment variables**
   ```bash
   cp .env.template .env
   # Edit .env and add your Google Cloud project ID
   ```

### Basic Usage

Translate a book:
```bash
python src/cli.py \
    --input "path/to/hindi-book.pdf" \
    --output "translated-book.docx" \
    --project-id YOUR_PROJECT_ID \
    --service-account service-account.json
```

With cost tracking:
```bash
python src/cli.py \
    --input "book.pdf" \
    --output "translation.docx" \
    --budget 100 \
    --alert-threshold 0.7
```

## Configuration ⚙️

### Translation Settings

Edit `config.json` to customize:

```json
{
  "translation": {
    "max_characters": 9000,
    "temperature": 0,
    "batch_size": 50
  },
  "cost": {
    "monthly_budget": 1000,
    "alert_threshold": 0.7,
    "price_per_1k_chars": 0.20
  }
}
```

### Cultural Glossary

The system includes a built-in glossary for common Hindi terms:
- श्री → Shri
- जी → -ji (honorific)
- पूजा → puja (worship ritual)

## Cost Information 💰

- **Pricing**: $0.20 per 1,000 characters
- **Average book cost**: ~$350 (without Mistral OCR)
- **With future Mistral OCR**: ~$90.50 (74% cost reduction)
- **Budget alerts**: Automatic warnings at 70% budget usage

## Technical Details 🔧

### Translation Limits
- Maximum 9,000 characters per request
- Maximum 256 sentences per batch
- Automatic chunking for large documents

### Quality Metrics
- MQM defect scoring (≤3 defects per 1,000 words)
- Glossary hit rate (≥98% required)
- Length ratio validation (0.8-1.5 acceptable)
- Back-translation similarity checks

### Supported Languages
- Source: Hindi (hi)
- Target: English (en)

## Development Status 📊

### Completed Features ✅
- [x] Google Cloud Translation integration
- [x] Multi-format document processing
- [x] Real-time cost tracking
- [x] PII scrubbing
- [x] Basic quality assurance
- [x] Translation memory

### In Progress 🚧
- [ ] CLI interface (Task 5)
- [ ] Main controller with batch processing (Task 6)
- [ ] Configuration system (Task 8)

### Future Enhancements 🔮
- [ ] Streamlit review dashboard
- [ ] Advanced error handling
- [ ] Progress tracking with tqdm
- [ ] Mistral OCR integration (when API becomes available)

## Testing 🧪

Run tests:
```bash
pytest tests/
```

Test with sample text:
```bash
python tests/test_sample.py
```

## Contributing 🤝

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙏

- Google Cloud Translation API for powering translations
- PyMuPDF for PDF processing
- The open-source community for various dependencies

## Support 💬

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check the [documentation](documentation/) folder
- Review the [setup instructions](documentation/setup-instructions.md)

---

Built with ❤️ for preserving and sharing Hindi literature globally 
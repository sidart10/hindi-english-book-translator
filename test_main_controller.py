#!/usr/bin/env python3
"""
Test script for the Main Controller
Demonstrates the batch processing and integration capabilities
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from main_controller import BookTranslationController


async def test_main_controller():
    """Test the main controller with a sample Hindi text"""
    
    print("=" * 70)
    print("🧪 TESTING MAIN CONTROLLER - HINDI TO ENGLISH TRANSLATION")
    print("=" * 70)
    
    # Create test input file
    test_input = "test_hindi_book.txt"
    test_output = "test_english_translation.docx"
    
    # Sample Hindi text from various sources
    hindi_text = """प्रस्तावना

यह एक परीक्षण पुस्तक है जो हिंदी से अंग्रेजी अनुवाद की क्षमता को प्रदर्शित करती है।

अध्याय १ - परिचय

भारत एक विविधतापूर्ण देश है। यहाँ विभिन्न भाषाएं, संस्कृतियां और परंपराएं हैं। हिंदी भारत की राष्ट्रभाषा है और करोड़ों लोग इसे बोलते हैं।

इस प्रणाली का उद्देश्य हिंदी पुस्तकों को अंग्रेजी में अनुवाद करना है ताकि वे वैश्विक दर्शकों तक पहुंच सकें। यह Google Cloud Translation API का उपयोग करता है।

अध्याय २ - तकनीकी विवरण

यह प्रणाली निम्नलिखित सुविधाएं प्रदान करती है:
- बैच प्रोसेसिंग के साथ दस्तावेज़ अनुवाद
- लागत ट्रैकिंग और बजट चेतावनी
- गुणवत्ता आश्वासन जांच
- DOCX आउटपुट जेनरेशन

प्रत्येक वाक्य को सावधानीपूर्वक अनुवादित किया जाता है। सिस्टम पृष्ठ संरचना को बनाए रखता है।

समापन

यह परीक्षण सफलतापूर्वक प्रदर्शित करता है कि सिस्टम कैसे काम करता है।"""
    
    # Write test file
    with open(test_input, 'w', encoding='utf-8') as f:
        f.write(hindi_text)
    
    print(f"\n📝 Created test input file: {test_input}")
    print(f"   Characters: {len(hindi_text)}")
    print(f"   Estimated cost: ${(len(hindi_text) / 1000) * 0.20:.4f}")
    
    try:
        # Initialize controller
        print("\n🚀 Initializing Main Controller...")
        controller = BookTranslationController("config.json")
        
        # Run translation
        print("\n📚 Starting translation process...")
        results = await controller.translate_book(
            test_input,
            test_output,
            project_id="demo_test"
        )
        
        # Display results
        print("\n" + "=" * 70)
        print("📊 TRANSLATION RESULTS")
        print("=" * 70)
        print(f"✅ Total Sentences: {results['statistics']['total_sentences']}")
        print(f"✅ Translated: {results['statistics']['translated_sentences']}")
        print(f"✅ Characters: {results['statistics']['total_characters']:,}")
        print(f"✅ Total Cost: ${results['statistics']['total_cost']:.4f}")
        
        if results['statistics']['errors']:
            print(f"\n⚠️  Errors encountered:")
            for error in results['statistics']['errors']:
                print(f"   - {error}")
        
        print(f"\n📄 Output saved to: {test_output}")
        
        # Show sample translation (if available)
        if Path(test_output).exists():
            print(f"   File size: {Path(test_output).stat().st_size:,} bytes")
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if Path(test_input).exists():
            os.remove(test_input)
            print(f"\n🧹 Cleaned up test input file")
    
    print("\n" + "=" * 70)
    print("Test complete!")


if __name__ == "__main__":
    # Set Google credentials if not already set
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        service_account_path = Path("service-account.json")
        if service_account_path.exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(service_account_path.absolute())
            print(f"✅ Set Google credentials: {service_account_path}")
        else:
            print("⚠️  Warning: service-account.json not found")
    
    # Run the test
    asyncio.run(test_main_controller()) 
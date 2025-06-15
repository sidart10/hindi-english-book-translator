#!/usr/bin/env python3
"""
Test script to verify Google Cloud Translation API setup
Tests Hindi to English translation with the Translation LLM
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Test Google Cloud setup
def test_google_cloud_setup():
    """Basic test to verify Google Cloud credentials"""
    print("Testing Google Cloud Setup...")
    
    try:
        from google.auth import default
        credentials, project = default()
        print(f"✅ Google Cloud authentication successful")
        print(f"   Project ID: {project}")
        return True
    except Exception as e:
        print(f"❌ Google Cloud authentication failed: {e}")
        return False

def test_translation_api():
    """Test the Translation API with sample Hindi text"""
    print("\nTesting Translation API...")
    
    try:
        from google.cloud import translate_v3 as translate
        
        # Initialize the client
        client = translate.TranslationServiceClient()
        
        # Get project ID
        project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
        if not project_id:
            # Try to get from credentials
            from google.auth import default
            _, project_id = default()
        
        if not project_id:
            print("❌ Project ID not found. Set GOOGLE_CLOUD_PROJECT environment variable.")
            return False
        
        # Test sentences
        test_sentences = [
            "नमस्ते, मेरा नाम राम है।",
            "मैं भारत से हूं।",
            "यह एक परीक्षण वाक्य है।",
            "हिंदी से अंग्रेजी अनुवाद का परीक्षण।"
        ]
        
        print(f"\nTranslating {len(test_sentences)} test sentences...")
        
        # Set up parent path
        parent = f"projects/{project_id}/locations/global"
        
        # Test each sentence
        for i, text in enumerate(test_sentences, 1):
            print(f"\n--- Test {i} ---")
            print(f"Hindi: {text}")
            
            # Prepare request
            request = translate.TranslateTextRequest(
                parent=parent,
                contents=[text],
                source_language_code="hi",
                target_language_code="en",
            )
            
            # Make the request
            response = client.translate_text(request=request)
            
            # Print translation
            for translation in response.translations:
                print(f"English: {translation.translated_text}")
        
        print("\n✅ Translation API test successful!")
        return True
        
    except ImportError:
        print("❌ google-cloud-translate not installed")
        print("   Run: pip install google-cloud-translate")
        return False
    except Exception as e:
        print(f"❌ Translation API test failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False

def test_character_limit():
    """Test the 9000 character limit"""
    print("\nTesting character limit handling...")
    
    # Create a long Hindi text (Lorem ipsum style)
    long_text = "यह एक बहुत लंबा परीक्षण पाठ है। " * 300  # ~9000 characters
    
    print(f"Testing with {len(long_text)} characters...")
    
    try:
        from google.cloud import translate_v3 as translate
        client = translate.TranslationServiceClient()
        
        project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
        if not project_id:
            from google.auth import default
            _, project_id = default()
        
        parent = f"projects/{project_id}/locations/global"
        
        # Try to translate
        request = translate.TranslateTextRequest(
            parent=parent,
            contents=[long_text[:8000]],  # Stay under 9k limit
            source_language_code="hi",
            target_language_code="en",
        )
        
        response = client.translate_text(request=request)
        
        print(f"✅ Successfully translated {len(long_text[:8000])} characters")
        print(f"   Translation preview: {response.translations[0].translated_text[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Character limit test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Google Cloud Translation API Test")
    print("=" * 50)
    
    # Check environment
    print("Environment Variables:")
    print(f"GOOGLE_APPLICATION_CREDENTIALS: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', 'Not set')}")
    print(f"GOOGLE_CLOUD_PROJECT: {os.environ.get('GOOGLE_CLOUD_PROJECT', 'Not set')}")
    
    # Run tests
    tests = [
        test_google_cloud_setup,
        test_translation_api,
        test_character_limit
    ]
    
    results = []
    for test in tests:
        print(f"\n{'=' * 50}")
        result = test()
        results.append(result)
    
    # Summary
    print(f"\n{'=' * 50}")
    print("TEST SUMMARY")
    print(f"{'=' * 50}")
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ All tests passed! Google Cloud setup is complete.")
        print("\nNext steps:")
        print("1. Update task status to 'done'")
        print("2. Proceed to Task 2: Create Core Translation Engine")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 
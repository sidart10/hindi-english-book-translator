#!/usr/bin/env python3
"""
Google Cloud Setup Verification Script
Verifies that all Google Cloud components are properly configured
"""

import os
import sys
import json
from pathlib import Path

def check_environment():
    """Check environment variables and credentials"""
    print("=== Checking Environment Variables ===")
    
    issues = []
    
    # Check GOOGLE_APPLICATION_CREDENTIALS
    creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if not creds_path:
        issues.append("❌ GOOGLE_APPLICATION_CREDENTIALS not set")
        print("❌ GOOGLE_APPLICATION_CREDENTIALS not set")
    else:
        print(f"✅ GOOGLE_APPLICATION_CREDENTIALS set to: {creds_path}")
        if not os.path.exists(creds_path):
            issues.append(f"❌ Service account file not found at: {creds_path}")
            print(f"❌ Service account file not found at: {creds_path}")
        else:
            print("✅ Service account file exists")
            
            # Verify it's valid JSON
            try:
                with open(creds_path, 'r') as f:
                    sa_data = json.load(f)
                    print(f"✅ Valid JSON service account file")
                    if 'project_id' in sa_data:
                        print(f"✅ Project ID in service account: {sa_data['project_id']}")
            except json.JSONDecodeError:
                issues.append("❌ Service account file is not valid JSON")
                print("❌ Service account file is not valid JSON")
    
    # Check GOOGLE_CLOUD_PROJECT
    project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
    if not project_id:
        # Try to get from service account file
        if creds_path and os.path.exists(creds_path):
            try:
                with open(creds_path, 'r') as f:
                    sa_data = json.load(f)
                    project_id = sa_data.get('project_id')
                    if project_id:
                        print(f"✅ Project ID found in service account: {project_id}")
                    else:
                        issues.append("❌ GOOGLE_CLOUD_PROJECT not set and not in service account")
                        print("❌ GOOGLE_CLOUD_PROJECT not set and not in service account")
            except:
                issues.append("❌ GOOGLE_CLOUD_PROJECT not set")
                print("❌ GOOGLE_CLOUD_PROJECT not set")
        else:
            issues.append("❌ GOOGLE_CLOUD_PROJECT not set")
            print("❌ GOOGLE_CLOUD_PROJECT not set")
    else:
        print(f"✅ GOOGLE_CLOUD_PROJECT set to: {project_id}")
    
    return issues

def check_gcloud_cli():
    """Check if gcloud CLI is installed"""
    print("\n=== Checking gcloud CLI ===")
    
    issues = []
    
    try:
        import subprocess
        result = subprocess.run(['gcloud', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ gcloud CLI is installed")
            print(f"   Version: {result.stdout.split()[2]}")
        else:
            issues.append("❌ gcloud CLI not working properly")
            print("❌ gcloud CLI not working properly")
    except FileNotFoundError:
        issues.append("❌ gcloud CLI not found. Please install it.")
        print("❌ gcloud CLI not found. Please install it.")
    
    return issues

def check_apis():
    """Check if required APIs can be imported"""
    print("\n=== Checking Python Libraries ===")
    
    issues = []
    
    # Check google-cloud-aiplatform
    try:
        import google.cloud.aiplatform as aiplatform
        print("✅ google-cloud-aiplatform is installed")
    except ImportError:
        issues.append("❌ google-cloud-aiplatform not installed")
        print("❌ google-cloud-aiplatform not installed")
        print("   Run: pip install google-cloud-aiplatform")
    
    # Check google-auth
    try:
        import google.auth
        print("✅ google-auth is installed")
    except ImportError:
        issues.append("❌ google-auth not installed")
        print("❌ google-auth not installed")
        print("   Run: pip install google-auth")
    
    return issues

def test_authentication():
    """Test Google Cloud authentication"""
    print("\n=== Testing Authentication ===")
    
    issues = []
    
    try:
        from google.auth import default
        from google.auth.exceptions import DefaultCredentialsError
        
        credentials, project = default()
        print("✅ Google Cloud authentication successful")
        print(f"✅ Default project: {project}")
        
    except DefaultCredentialsError as e:
        issues.append(f"❌ Authentication failed: {str(e)}")
        print(f"❌ Authentication failed: {str(e)}")
    except Exception as e:
        issues.append(f"❌ Unexpected error: {str(e)}")
        print(f"❌ Unexpected error: {str(e)}")
    
    return issues

def create_env_template():
    """Create .env.template file"""
    env_template = """# Google Cloud Configuration
GOOGLE_APPLICATION_CREDENTIALS=./service-account.json
GOOGLE_CLOUD_PROJECT=your-project-id-here
"""
    
    with open('.env.template', 'w') as f:
        f.write(env_template)
    
    print("\n✅ Created .env.template file")

def main():
    """Run all verification checks"""
    print("Google Cloud Setup Verification")
    print("=" * 50)
    
    all_issues = []
    
    # Run checks
    all_issues.extend(check_environment())
    all_issues.extend(check_gcloud_cli())
    all_issues.extend(check_apis())
    all_issues.extend(test_authentication())
    
    # Create template files
    create_env_template()
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    if not all_issues:
        print("✅ All checks passed! Your Google Cloud setup is ready.")
        print("\nNext steps:")
        print("1. Run test_sample.py to test translation")
        print("2. Start implementing the translation engine")
    else:
        print(f"❌ Found {len(all_issues)} issue(s):")
        for issue in all_issues:
            print(f"   - {issue}")
        
        print("\nPlease fix these issues and run this script again.")
        sys.exit(1)

if __name__ == "__main__":
    main() 
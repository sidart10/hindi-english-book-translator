# Google Cloud Setup Scripts

This directory contains scripts to help you set up Google Cloud for the Hindi-English Book Translation System.

## Quick Start

### Automated Setup (Recommended)

Run the automated setup script:

```bash
./setup_gcp.sh
```

This script will:
- Check for gcloud CLI installation
- Enable required Google Cloud APIs
- Create a service account with necessary permissions
- Download service account credentials
- Create a `.env` file with configuration

### Manual Setup

If you prefer manual setup or the script fails, follow the detailed guide:
- [Google Cloud Setup Guide](../docs/google-cloud-setup.md)

### Verification

After setup, verify everything is working:

```bash
# Check Google Cloud configuration
python verify_gcp_setup.py

# Test translation functionality
python ../tests/test_sample.py
```

## Files in this Directory

- `setup_gcp.sh` - Automated setup script
- `verify_gcp_setup.py` - Verification script to check your setup
- `README.md` - This file

## Prerequisites

1. **Google Cloud Account** with billing enabled
2. **gcloud CLI** installed:
   - macOS: `brew install --cask google-cloud-sdk`
   - Windows/Linux: [Download from Google](https://cloud.google.com/sdk/docs/install)
3. **Python 3.8+** installed

## Common Issues

### "Permission denied" when running setup_gcp.sh
Make sure the script is executable:
```bash
chmod +x setup_gcp.sh
```

### "API not enabled" errors
The script should enable APIs automatically, but if it fails, enable them manually:
```bash
gcloud services enable aiplatform.googleapis.com
gcloud services enable translate.googleapis.com
```

### "Billing account not linked"
You need to enable billing on your Google Cloud project. Visit the [Google Cloud Console](https://console.cloud.google.com/billing).

## Next Steps

After successful setup:
1. Install Python dependencies: `pip install -r requirements.txt`
2. Start implementing the translation engine (Task 2) 
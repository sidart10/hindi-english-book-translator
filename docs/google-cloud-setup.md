# Google Cloud Setup Guide for Hindi-English Book Translation System

This guide will help you set up Google Cloud for the Hindi-English Book Translation System using Google's Translation LLM.

## Prerequisites

- Google account
- Credit card for Google Cloud billing (you get $300 free credits for new accounts)
- Terminal/Command line access

## Step 1: Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click on the project dropdown at the top
3. Click "New Project"
4. Enter project details:
   - Project name: `book-translator` (or your preferred name)
   - Project ID: Will be auto-generated (note this for later)
5. Click "Create"

## Step 2: Enable Required APIs

Run these commands in your terminal after installing gcloud CLI (see Step 4):

```bash
# Set your project ID
export PROJECT_ID="gen-lang-client-0753897804"

# Set the project
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable translate.googleapis.com
```

Or enable manually in the Console:
1. Go to APIs & Services → Library
2. Search for and enable:
   - Vertex AI API
   - Cloud Translation API

## Step 3: Create Service Account

1. In Cloud Console, go to IAM & Admin → Service Accounts
2. Click "Create Service Account"
3. Enter details:
   - Service account name: `book-translator`
   - Service account ID: `book-translator`
   - Description: "Service account for book translation system"
4. Click "Create and Continue"
5. Add roles:
   - Click "Add Role"
   - Search and select: `Vertex AI User`
   - Add another role: `Cloud Translation API User`
6. Click "Continue" then "Done"

## Step 4: Download Service Account Key

1. Click on the created service account
2. Go to "Keys" tab
3. Click "Add K 

ey" → "Create new key"
4. Select "JSON" format
5. Click "Create"
6. Save the downloaded file as `service-account.json` in your project root
7. **IMPORTANT**: Add `service-account.json` to `.gitignore`

## Step 5: Install gcloud CLI

### macOS
```bash
# Using Homebrew
brew install --cask google-cloud-sdk

# Or download from: https://cloud.google.com/sdk/docs/install-sdk#mac
```

### Windows
Download installer from: https://cloud.google.com/sdk/docs/install-sdk#windows

### Linux
```bash
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar -xf google-cloud-cli-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh
```

## Step 6: Configure Authentication

```bash
# Initialize gcloud
gcloud init

# Authenticate with your Google account
gcloud auth login

# Set project
gcloud config set project $PROJECT_ID

# Set application default credentials
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/service-account.json"
```

## Step 7: Set Environment Variables

Create a `.env` file in your project root:

```bash
# .env
GOOGLE_APPLICATION_CREDENTIALS=./service-account.json
GOOGLE_CLOUD_PROJECT=book-translator-463006
```

## Step 8: Verify Setup

Run the verification script:

```bash
python setup/verify_gcp_setup.py
```

## Troubleshooting

### "API not enabled" Error
Make sure you've enabled both:
- Vertex AI API
- Cloud Translation API

### "Permission denied" Error
Ensure your service account has these roles:
- Vertex AI User
- Cloud Translation API User

### "Credentials not found" Error
Check that:
- `service-account.json` exists in the correct location
- `GOOGLE_APPLICATION_CREDENTIALS` environment variable is set correctly
- The path to the JSON file is absolute or relative to where you run the script

## Cost Considerations

- Translation API pricing: ~$20 per million characters
- For our use case (books): Approximately $0.20 per 1000 characters
- Set up billing alerts at 70% of your budget
- Monitor usage in Cloud Console → Billing

## Next Steps

After completing this setup:
1. Run `test_sample.py` to verify the translation works
2. Proceed to Task 2: Create Core Translation Engine 
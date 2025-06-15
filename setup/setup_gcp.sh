#!/bin/bash

# Google Cloud Setup Script for Book Translator
# This script helps automate the Google Cloud setup process

set -e

echo "Google Cloud Setup for Book Translator"
echo "======================================"
echo

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Check if gcloud is installed
echo "Checking for gcloud CLI..."
if command -v gcloud &> /dev/null; then
    print_success "gcloud CLI is installed"
    gcloud version
else
    print_error "gcloud CLI not found"
    echo "Please install gcloud CLI first:"
    echo "  macOS: brew install --cask google-cloud-sdk"
    echo "  Or visit: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Get current project
CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null)
if [ -z "$CURRENT_PROJECT" ]; then
    print_warning "No project currently set"
else
    echo "Current project: $CURRENT_PROJECT"
fi

# Ask for project ID
echo
read -p "Enter your Google Cloud Project ID (or press Enter to use $CURRENT_PROJECT): " PROJECT_ID
PROJECT_ID=${PROJECT_ID:-$CURRENT_PROJECT}

if [ -z "$PROJECT_ID" ]; then
    print_error "Project ID is required"
    exit 1
fi

echo
echo "Using project: $PROJECT_ID"

# Set the project
echo "Setting project..."
gcloud config set project $PROJECT_ID
print_success "Project set to $PROJECT_ID"

# Enable required APIs
echo
echo "Enabling required APIs..."
echo "This may take a few minutes..."

# Enable Vertex AI API
echo "Enabling Vertex AI API..."
if gcloud services enable aiplatform.googleapis.com --project=$PROJECT_ID; then
    print_success "Vertex AI API enabled"
else
    print_error "Failed to enable Vertex AI API"
fi

# Enable Translation API
echo "Enabling Translation API..."
if gcloud services enable translate.googleapis.com --project=$PROJECT_ID; then
    print_success "Translation API enabled"
else
    print_error "Failed to enable Translation API"
fi

# Create service account
echo
echo "Creating service account..."
SERVICE_ACCOUNT_NAME="book-translator"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# Check if service account already exists
if gcloud iam service-accounts describe $SERVICE_ACCOUNT_EMAIL --project=$PROJECT_ID &>/dev/null; then
    print_warning "Service account already exists"
else
    if gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME \
        --display-name="Book Translator Service Account" \
        --project=$PROJECT_ID; then
        print_success "Service account created"
    else
        print_error "Failed to create service account"
    fi
fi

# Grant roles to service account
echo
echo "Granting roles to service account..."

# Grant Vertex AI User role
echo "Granting Vertex AI User role..."
if gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/aiplatform.user" \
    --quiet; then
    print_success "Vertex AI User role granted"
else
    print_error "Failed to grant Vertex AI User role"
fi

# Grant Translation API User role
echo "Granting Cloud Translation API User role..."
if gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/cloudtranslate.user" \
    --quiet; then
    print_success "Cloud Translation API User role granted"
else
    print_error "Failed to grant Cloud Translation API User role"
fi

# Create service account key
echo
echo "Creating service account key..."
KEY_FILE="service-account.json"

if [ -f "$KEY_FILE" ]; then
    print_warning "service-account.json already exists"
    read -p "Do you want to create a new key? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping key creation"
    else
        if gcloud iam service-accounts keys create $KEY_FILE \
            --iam-account=$SERVICE_ACCOUNT_EMAIL \
            --project=$PROJECT_ID; then
            print_success "Service account key created: $KEY_FILE"
        else
            print_error "Failed to create service account key"
        fi
    fi
else
    if gcloud iam service-accounts keys create $KEY_FILE \
        --iam-account=$SERVICE_ACCOUNT_EMAIL \
        --project=$PROJECT_ID; then
        print_success "Service account key created: $KEY_FILE"
    else
        print_error "Failed to create service account key"
    fi
fi

# Create .env file
echo
echo "Creating .env file..."
if [ -f ".env" ]; then
    print_warning ".env file already exists"
else
    cat > .env << EOF
# Google Cloud Configuration
GOOGLE_APPLICATION_CREDENTIALS=./service-account.json
GOOGLE_CLOUD_PROJECT=$PROJECT_ID
EOF
    print_success ".env file created"
fi

# Update .gitignore
echo
echo "Updating .gitignore..."
if ! grep -q "service-account.json" .gitignore 2>/dev/null; then
    echo "service-account.json" >> .gitignore
    print_success "Added service-account.json to .gitignore"
else
    print_success "service-account.json already in .gitignore"
fi

if ! grep -q ".env" .gitignore 2>/dev/null; then
    echo ".env" >> .gitignore
    print_success "Added .env to .gitignore"
else
    print_success ".env already in .gitignore"
fi

# Final instructions
echo
echo "======================================"
print_success "Google Cloud setup complete!"
echo "======================================"
echo
echo "Next steps:"
echo "1. Install Python dependencies:"
echo "   pip install google-cloud-aiplatform google-cloud-translate"
echo
echo "2. Export environment variables:"
echo "   export GOOGLE_APPLICATION_CREDENTIALS=./service-account.json"
echo "   export GOOGLE_CLOUD_PROJECT=$PROJECT_ID"
echo
echo "3. Run the verification script:"
echo "   python setup/verify_gcp_setup.py"
echo
echo "4. Run the test script:"
echo "   python tests/test_sample.py" 
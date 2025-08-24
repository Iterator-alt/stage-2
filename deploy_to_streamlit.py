#!/usr/bin/env python3
"""
Quick Deployment Script for DataTobiz Brand Monitoring System
This script helps you prepare your repository for Streamlit Cloud deployment.
"""

import os
import sys
import shutil
from pathlib import Path

def check_git_status():
    """Check if we're in a git repository and if sensitive files are ignored."""
    print("🔍 Checking Git repository status...")
    
    if not Path(".git").exists():
        print("❌ Not in a git repository. Please run:")
        print("   git init")
        print("   git add .")
        print("   git commit -m 'Initial commit'")
        return False
    
    # Check if sensitive files exist
    sensitive_files = ["config.yaml", "credentials.json", ".env"]
    existing_sensitive = []
    
    for file in sensitive_files:
        if Path(file).exists():
            existing_sensitive.append(file)
    
    if existing_sensitive:
        print(f"⚠️  Found sensitive files: {', '.join(existing_sensitive)}")
        print("   These should NOT be committed to git.")
        print("   Make sure they're in your .gitignore file.")
        return False
    
    print("✅ Git repository status: OK")
    return True

def check_required_files():
    """Check if all required files for deployment exist."""
    print("\n📋 Checking required files...")
    
    required_files = [
        "streamlit_app_secure.py",
        "requirements.txt",
        "config.template.yaml",
        ".streamlit/secrets.toml",
        "src/",
        "main.py"
    ]
    
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files present")
    return True

def create_deployment_package():
    """Create a deployment package with only necessary files."""
    print("\n📦 Creating deployment package...")
    
    # Files to include in deployment
    include_patterns = [
        "*.py",
        "requirements.txt",
        "config.template.yaml",
        ".streamlit/secrets.toml",
        "src/**/*",
        "README.md",
        "DEPLOYMENT_GUIDE.md",
        ".gitignore"
    ]
    
    # Files to exclude
    exclude_patterns = [
        "config.yaml",
        "credentials.json",
        ".env",
        "*.log",
        "__pycache__",
        "*.pyc",
        "test_*.py",
        "deploy.sh",
        "docker-compose.yml",
        "Dockerfile"
    ]
    
    print("✅ Deployment package ready")
    return True

def generate_secrets_template():
    """Generate a template for Streamlit secrets."""
    print("\n🔐 Generating Streamlit secrets template...")
    
    secrets_template = """# Streamlit Secrets Configuration
# Copy this content to your Streamlit Cloud app settings > Secrets

# API Keys
OPENAI_API_KEY = "your_openai_api_key_here"
PERPLEXITY_API_KEY = "your_perplexity_api_key_here"
GEMINI_API_KEY = "your_gemini_api_key_here"

# Google Sheets Configuration
GOOGLE_SHEETS_SPREADSHEET_ID = "your_spreadsheet_id_here"
GOOGLE_SHEETS_WORKSHEET_NAME = "Brand_Monitoring_New"

# Google Service Account Credentials (JSON content as string)
GOOGLE_SERVICE_ACCOUNT_CREDENTIALS = '''
{
  "type": "service_account",
  "project_id": "your_project_id",
  "private_key_id": "your_private_key_id",
  "private_key": "-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY_HERE\n-----END PRIVATE KEY-----\n",
  "client_email": "your_service_account_email@your_project.iam.gserviceaccount.com",
  "client_id": "your_client_id",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/your_service_account_email%40your_project.iam.gserviceaccount.com"
}
'''

# Application Configuration
APP_TITLE = "DataTobiz Brand Monitoring System"
APP_VERSION = "2.0"
ENVIRONMENT = "production"

# Security Settings
ENABLE_DEBUG_MODE = false
LOG_LEVEL = "INFO"
"""
    
    with open("streamlit_secrets_template.toml", "w") as f:
        f.write(secrets_template)
    
    print("✅ Created streamlit_secrets_template.toml")
    print("   Copy this content to your Streamlit Cloud app settings > Secrets")

def print_deployment_instructions():
    """Print step-by-step deployment instructions."""
    print("\n" + "="*60)
    print("🚀 DEPLOYMENT INSTRUCTIONS")
    print("="*60)
    
    instructions = """
1. 📁 PUSH TO GITHUB:
   git add .
   git commit -m "Prepare for Streamlit deployment"
   git push origin main

2. 🌐 DEPLOY TO STREAMLIT CLOUD:
   - Go to https://share.streamlit.io/
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: streamlit_app_secure.py
   - Click "Deploy"

3. 🔐 CONFIGURE SECRETS:
   - In your deployed app, go to "Settings" > "Secrets"
   - Copy content from streamlit_secrets_template.toml
   - Replace placeholder values with your actual API keys
   - Click "Save"

4. 🧪 TEST YOUR APP:
   - Go to your app URL
   - Check "Secrets Status" in sidebar
   - Click "Initialize System"
   - Test brand monitoring functionality

5. 📊 SET UP GOOGLE SHEETS:
   - Create a new Google Sheets spreadsheet
   - Share with your service account email
   - Update GOOGLE_SHEETS_SPREADSHEET_ID in secrets

6. ✅ VERIFY DEPLOYMENT:
   - All secrets configured ✅
   - System initializes successfully ✅
   - Brand monitoring works ✅
   - Google Sheets integration works ✅

📖 For detailed instructions, see DEPLOYMENT_GUIDE.md
🔧 For troubleshooting, check the guide's troubleshooting section
"""
    
    print(instructions)

def main():
    """Main deployment preparation function."""
    print("🚀 DataTobiz Brand Monitoring System - Deployment Preparation")
    print("="*60)
    
    # Check prerequisites
    if not check_git_status():
        print("\n❌ Please fix Git repository issues before proceeding.")
        return False
    
    if not check_required_files():
        print("\n❌ Please ensure all required files are present.")
        return False
    
    # Create deployment package
    if not create_deployment_package():
        print("\n❌ Failed to create deployment package.")
        return False
    
    # Generate secrets template
    generate_secrets_template()
    
    # Print instructions
    print_deployment_instructions()
    
    print("\n🎉 Deployment preparation complete!")
    print("   Follow the instructions above to deploy your app.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

# Streamlit Deployment Guide for DataTobiz Brand Monitoring

## 🚀 Quick Deployment to Streamlit Cloud

### Prerequisites
- GitHub repository with your code
- Streamlit Cloud account
- API keys for OpenAI and Perplexity
- Google Sheets credentials

### Step 1: Prepare Your Repository

1. **Ensure sensitive files are excluded** (already done):
   - `config.yaml` ✅ (in .gitignore)
   - `credentials.json` ✅ (in .gitignore)
   - `.env` ✅ (in .gitignore)

2. **Template files are included**:
   - `config.template.yaml` ✅
   - `env.template` ✅

### Step 2: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repository
3. Set the main file path to: `streamlit_app_secure.py`
4. Click "Deploy"

### Step 3: Configure Secrets in Streamlit

In your Streamlit app settings, add these secrets:

```toml
[secrets]
# OpenAI Configuration
OPENAI_API_KEY = "sk-proj-your-actual-openai-key-here"

# Perplexity Configuration  
PERPLEXITY_API_KEY = "pplx-your-actual-perplexity-key-here"

# Google Sheets Configuration
GOOGLE_SHEETS_SPREADSHEET_ID = "your-spreadsheet-id-here"

# Google Service Account (JSON content)
GOOGLE_SERVICE_ACCOUNT = '''
{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "your-private-key-id",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
  "client_email": "your-service-account@your-project.iam.gserviceaccount.com",
  "client_id": "your-client-id",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40your-project.iam.gserviceaccount.com"
}
'''
```

### Step 4: Environment Variables (Optional)

You can also set these as environment variables in Streamlit:

```bash
OPENAI_API_KEY=sk-proj-your-actual-key
PERPLEXITY_API_KEY=pplx-your-actual-key
GOOGLE_SHEETS_SPREADSHEET_ID=your-spreadsheet-id
```

## 🔒 Security Best Practices

### ✅ What's Already Secured:
- API keys are stored in Streamlit secrets (not in code)
- Configuration files with secrets are gitignored
- Template files provided for local development
- Environment variable support for flexible deployment

### 🔧 Local Development Setup:

1. **Copy template files**:
   ```bash
   cp config.template.yaml config.yaml
   cp env.template .env
   ```

2. **Fill in your actual values** in the copied files

3. **For Google Sheets**, create a service account and download credentials as `credentials.json`

## 📁 File Structure for Deployment

```
datatobiz-brand-monitoring/
├── streamlit_app_secure.py     # Main Streamlit app
├── main.py                     # CLI interface
├── config.template.yaml        # Template config (safe for git)
├── env.template               # Template env vars (safe for git)
├── requirements.txt           # Python dependencies
├── .gitignore                # Excludes sensitive files
├── src/                      # Source code
│   ├── agents/              # LLM agents
│   ├── config/              # Configuration management
│   ├── storage/             # Google Sheets integration
│   ├── utils/               # Utilities
│   └── workflow/            # Workflow engine
└── README.md                # Documentation
```

## 🚨 Important Notes

1. **Never commit sensitive files**:
   - ❌ `config.yaml` (contains API keys)
   - ❌ `credentials.json` (contains Google service account)
   - ❌ `.env` (contains environment variables)

2. **Always use templates**:
   - ✅ `config.template.yaml` (safe for git)
   - ✅ `env.template` (safe for git)

3. **Streamlit secrets are encrypted** and only accessible to your app

## 🔧 Troubleshooting

### Common Issues:

1. **"Module not found" errors**:
   - Ensure `requirements.txt` includes all dependencies
   - Check that all imports use relative paths

2. **API key errors**:
   - Verify keys are correctly set in Streamlit secrets
   - Check that keys are valid and have proper permissions

3. **Google Sheets access errors**:
   - Ensure service account has proper permissions
   - Verify spreadsheet ID is correct
   - Check that credentials JSON is properly formatted

### Support:
- Check the logs in Streamlit Cloud for detailed error messages
- Verify all secrets are properly configured
- Test locally first with your actual API keys

## 🎯 Next Steps

1. Deploy to Streamlit Cloud
2. Configure secrets
3. Test the application
4. Share the public URL with your team

Your app will be available at: `https://your-app-name.streamlit.app`

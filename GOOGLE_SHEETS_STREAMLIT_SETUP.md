# Google Sheets Integration Setup for Streamlit

This guide will help you set up Google Sheets integration to work properly with the DataTobiz Brand Monitoring System in Streamlit.

## 🎯 Overview

The system now supports two ways to configure Google Sheets:

1. **Streamlit Secrets (Recommended for deployment)**: Store credentials securely in Streamlit Cloud
2. **Local Configuration**: Use local `config.yaml` and `credentials.json` files

## 🔧 Setup Methods

### Method 1: Streamlit Secrets (Recommended)

#### Step 1: Create Google Service Account

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable Google Sheets API and Google Drive API
4. Create a Service Account:
   - Go to "IAM & Admin" > "Service Accounts"
   - Click "Create Service Account"
   - Give it a name (e.g., "datatobiz-brand-monitoring")
   - Grant "Editor" role
   - Create and download the JSON key file

#### Step 2: Set up Google Sheets

1. Create a new Google Sheet or use existing one
2. Share the sheet with your service account email (found in the JSON file)
3. Give it "Editor" permissions
4. Copy the Spreadsheet ID from the URL:
   ```
   https://docs.google.com/spreadsheets/d/YOUR_SPREADSHEET_ID/edit
   ```

#### Step 3: Configure Streamlit Secrets

In your Streamlit Cloud deployment, add these secrets:

```toml
# .streamlit/secrets.toml
OPENAI_API_KEY = "your-openai-api-key"
PERPLEXITY_API_KEY = "your-perplexity-api-key"
GEMINI_API_KEY = "your-gemini-api-key"
GOOGLE_SHEETS_SPREADSHEET_ID = "your-spreadsheet-id"
GOOGLE_SERVICE_ACCOUNT_CREDENTIALS = '''
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

**Important**: Copy the entire JSON content from your downloaded service account file into the `GOOGLE_SERVICE_ACCOUNT_CREDENTIALS` secret.

### Method 2: Local Configuration

#### Step 1: Prepare Files

1. **credentials.json**: Place your Google service account JSON file in the project root
2. **config.yaml**: Update the Google Sheets configuration:

```yaml
google_sheets:
  spreadsheet_id: "your-spreadsheet-id"
  worksheet_name: "Brand_Monitoring_New"
  credentials_file: "credentials.json"
```

## 🧪 Testing the Setup

### Option 1: Use the Test Script

Run the provided test script to verify your setup:

```bash
python test_google_sheets_streamlit.py
```

This script will:
- Check if credentials are properly configured
- Test Google Sheets connection
- Verify data access and retrieval
- Display connection details

### Option 2: Test in Streamlit App

1. Start the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

2. Go to the "Historical Data" tab
3. Click "🧪 Test Google Sheets Connection"
4. Check the sidebar for Google Sheets status

## 🔍 Troubleshooting

### Common Issues

#### 1. "Credentials file not found"
- **Solution**: Ensure `credentials.json` is in the project root
- **For Streamlit**: Check that `GOOGLE_SERVICE_ACCOUNT_CREDENTIALS` secret is properly formatted

#### 2. "Spreadsheet not found"
- **Solution**: Verify the spreadsheet ID is correct
- **Check**: The ID is the long string in the Google Sheets URL

#### 3. "Permission denied"
- **Solution**: Share the spreadsheet with your service account email
- **Required**: Editor permissions for the service account

#### 4. "API not enabled"
- **Solution**: Enable Google Sheets API and Google Drive API in Google Cloud Console

### Debug Steps

1. **Check Configuration**:
   ```python
   # In Streamlit app sidebar, click "📋 View Config"
   # Verify Google Sheets ID is displayed
   ```

2. **Test Connection**:
   ```python
   # In Historical Data tab, click "🧪 Test Google Sheets Connection"
   # Check for detailed error messages
   ```

3. **Verify Permissions**:
   - Service account has Editor access to the spreadsheet
   - APIs are enabled in Google Cloud Console
   - Spreadsheet ID is correct

## 📊 Expected Behavior

When properly configured, you should see:

### In Streamlit Sidebar:
- ✅ Google Sheets Connected
- 📊 X records found

### In Historical Data Tab:
- Connection test shows success
- Spreadsheet and worksheet details displayed
- Record count and column count shown

### In System Health Tab:
- Google Sheets shows as healthy
- Storage status indicates connected

## 🚀 Deployment Checklist

Before deploying to Streamlit Cloud:

- [ ] Google Service Account created
- [ ] Google Sheets API enabled
- [ ] Spreadsheet shared with service account
- [ ] Streamlit secrets configured
- [ ] Test script passes locally
- [ ] Connection test works in Streamlit

## 🔒 Security Notes

- Never commit `credentials.json` to version control
- Use Streamlit secrets for production deployments
- Regularly rotate service account keys
- Limit service account permissions to minimum required

## 📞 Support

If you encounter issues:

1. Run the test script for detailed diagnostics
2. Check the troubleshooting section above
3. Verify all configuration steps were completed
4. Ensure APIs are enabled in Google Cloud Console

## 🎉 Success Indicators

You'll know Google Sheets is working when:

- ✅ Connection test passes
- ✅ Historical data loads
- ✅ New monitoring results are stored
- ✅ Analytics dashboard shows data
- ✅ No error messages in Streamlit

The system will automatically create the worksheet with proper headers and store all monitoring results with enhanced analytics data.

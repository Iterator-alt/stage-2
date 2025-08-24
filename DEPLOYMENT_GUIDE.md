# 🚀 DataTobiz Brand Monitoring System - Global Deployment Guide

## 📋 **Overview**

This guide provides step-by-step instructions for deploying the DataTobiz Brand Monitoring System globally on Streamlit Cloud with secure secrets management.

## 🔐 **Security Strategy**

### **What Gets Pushed to Git:**
- ✅ Source code (Python files)
- ✅ Requirements and dependencies
- ✅ Template configuration files
- ✅ Documentation and guides
- ✅ Streamlit app files

### **What Stays Private (Streamlit Secrets):**
- ❌ API Keys (OpenAI, Perplexity, Gemini)
- ❌ Google Service Account credentials
- ❌ Google Sheets configuration
- ❌ Production configuration files

## 🛠️ **Step-by-Step Deployment Process**

### **Step 1: Prepare Your Repository**

1. **Create a new GitHub repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: DataTobiz Brand Monitoring System"
   git branch -M main
   git remote add origin https://github.com/yourusername/datatobiz-brand-monitoring.git
   git push -u origin main
   ```

2. **Verify .gitignore is working**
   - Check that `config.yaml` and `credentials.json` are NOT in your repository
   - Only template files should be committed

### **Step 2: Set Up Google Cloud Project**

1. **Create a Google Cloud Project**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing one

2. **Enable Google Sheets API**
   - Go to "APIs & Services" > "Library"
   - Search for "Google Sheets API"
   - Click "Enable"

3. **Create Service Account**
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "Service Account"
   - Fill in details and create
   - Download the JSON key file

4. **Create Google Sheets**
   - Go to [Google Sheets](https://sheets.google.com/)
   - Create a new spreadsheet
   - Share it with your service account email
   - Copy the spreadsheet ID from the URL

### **Step 3: Get API Keys**

1. **OpenAI API Key**
   - Go to [OpenAI Platform](https://platform.openai.com/api-keys)
   - Create a new API key
   - Copy the key (starts with `sk-`)

2. **Perplexity API Key**
   - Go to [Perplexity AI](https://www.perplexity.ai/settings/api)
   - Create a new API key
   - Copy the key (starts with `pplx-`)

3. **Google Gemini API Key**
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the key

### **Step 4: Deploy to Streamlit Cloud**

1. **Sign up for Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io/)
   - Sign in with GitHub

2. **Deploy Your App**
   - Click "New app"
   - Select your repository
   - Set main file path: `streamlit_app_secure.py`
   - Click "Deploy"

3. **Configure Secrets**
   - In your deployed app, go to "Settings" > "Secrets"
   - Add the following secrets:

   ```toml
   # API Keys
   OPENAI_API_KEY = "sk-your-openai-key-here"
   PERPLEXITY_API_KEY = "pplx-your-perplexity-key-here"
   GEMINI_API_KEY = "your-gemini-key-here"
   
   # Google Sheets Configuration
   GOOGLE_SHEETS_SPREADSHEET_ID = "your-spreadsheet-id-here"
   GOOGLE_SHEETS_WORKSHEET_NAME = "Brand_Monitoring_New"
   
   # Google Service Account Credentials (JSON content as string)
   GOOGLE_SERVICE_ACCOUNT_CREDENTIALS = '''
   {
     "type": "service_account",
     "project_id": "your-project-id",
     "private_key_id": "your-private-key-id",
     "private_key": "-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY_HERE\n-----END PRIVATE KEY-----\n",
     "client_email": "your-service-account@your-project.iam.gserviceaccount.com",
     "client_id": "your-client-id",
     "auth_uri": "https://accounts.google.com/o/oauth2/auth",
     "token_uri": "https://oauth2.googleapis.com/token",
     "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
     "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40your-project.iam.gserviceaccount.com"
   }
   '''
   
   # Application Configuration
   APP_TITLE = "DataTobiz Brand Monitoring System"
   APP_VERSION = "2.0"
   ENVIRONMENT = "production"
   
   # Security Settings
   ENABLE_DEBUG_MODE = false
   LOG_LEVEL = "INFO"
   ```

4. **Redeploy the App**
   - After adding secrets, click "Redeploy"
   - Your app will now have access to the secure configuration

### **Step 5: Test Your Deployment**

1. **Access Your App**
   - Your app will be available at: `https://your-app-name-your-username.streamlit.app`

2. **Verify Secrets Status**
   - Check the sidebar for "Secrets Status"
   - Should show "✅ All secrets configured"

3. **Initialize System**
   - Click "🚀 Initialize System" in the sidebar
   - Should show "✅ System initialized successfully!"

4. **Test Brand Monitoring**
   - Enter a test query like "best data analytics companies"
   - Click "🚀 Start Monitoring"
   - Verify results are displayed correctly

## 🔧 **Troubleshooting**

### **Common Issues:**

1. **"Missing secrets" error**
   - Double-check all secrets are added correctly
   - Ensure no extra spaces or quotes
   - Verify JSON format for Google credentials

2. **"Failed to initialize system"**
   - Check API keys are valid
   - Verify Google Sheets permissions
   - Check logs for specific error messages

3. **"Google Sheets not accessible"**
   - Ensure service account has edit permissions
   - Verify spreadsheet ID is correct
   - Check Google Sheets API is enabled

4. **"Agent initialization failed"**
   - Verify API keys are correct
   - Check API quotas and limits
   - Ensure models are available

### **Debug Mode:**
- Set `ENABLE_DEBUG_MODE = true` in secrets for detailed error messages
- Check Streamlit Cloud logs for additional information

## 📊 **Monitoring and Maintenance**

### **Regular Checks:**
1. **API Usage Monitoring**
   - Monitor API usage in respective platforms
   - Set up billing alerts
   - Track cost trends

2. **System Health**
   - Use the "System Health" tab in your app
   - Run health checks regularly
   - Monitor agent performance

3. **Data Quality**
   - Review Google Sheets data regularly
   - Check for missing or incorrect entries
   - Validate brand detection accuracy

### **Updates and Maintenance:**
1. **Code Updates**
   - Push changes to GitHub
   - Streamlit Cloud will auto-deploy
   - Test thoroughly after updates

2. **Configuration Changes**
   - Update secrets in Streamlit Cloud
   - No need to redeploy code
   - Changes take effect immediately

3. **API Key Rotation**
   - Generate new API keys
   - Update secrets in Streamlit Cloud
   - Test functionality

## 🔒 **Security Best Practices**

### **API Key Security:**
- ✅ Use environment-specific keys
- ✅ Rotate keys regularly
- ✅ Monitor usage patterns
- ✅ Set up billing alerts

### **Access Control:**
- ✅ Limit Google Sheets permissions
- ✅ Use service accounts for automation
- ✅ Regular access reviews
- ✅ Monitor for suspicious activity

### **Data Protection:**
- ✅ Encrypt sensitive data
- ✅ Regular backups
- ✅ Access logging
- ✅ Compliance monitoring

## 🌐 **Global Deployment Considerations**

### **Performance:**
- Streamlit Cloud provides global CDN
- Automatic scaling based on usage
- No server management required

### **Cost Optimization:**
- Monitor API usage and costs
- Set up usage limits
- Optimize query patterns
- Use appropriate model tiers

### **Reliability:**
- Streamlit Cloud provides 99.9% uptime
- Automatic failover
- Built-in monitoring
- Regular maintenance windows

## 📞 **Support and Resources**

### **Documentation:**
- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)
- [Google Sheets API Guide](https://developers.google.com/sheets/api)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Perplexity API Documentation](https://docs.perplexity.ai/)

### **Community:**
- [Streamlit Community](https://discuss.streamlit.io/)
- [GitHub Issues](https://github.com/your-repo/issues)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/streamlit)

## 🎉 **Success Checklist**

- [ ] Repository created and code pushed
- [ ] Google Cloud project set up
- [ ] API keys obtained and secured
- [ ] Streamlit Cloud app deployed
- [ ] Secrets configured correctly
- [ ] System initialization successful
- [ ] Brand monitoring tested
- [ ] Google Sheets integration working
- [ ] Health checks passing
- [ ] Documentation updated

## 🚀 **Next Steps**

1. **Customization**
   - Modify brand variations
   - Adjust detection parameters
   - Customize UI/UX

2. **Scaling**
   - Add more agents
   - Implement caching
   - Optimize performance

3. **Integration**
   - Connect to other data sources
   - Add notification systems
   - Implement automated reporting

4. **Analytics**
   - Set up advanced analytics
   - Create custom dashboards
   - Implement trend analysis

---

**🎯 Your DataTobiz Brand Monitoring System is now globally deployed and ready for production use!**

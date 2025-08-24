# DataTobiz Brand Monitoring System - Deployment Guide

## 🚀 Quick Start

### Prerequisites

1. **Docker** and **Docker Compose** installed
2. **config.yaml** file with your API keys
3. **credentials.json** file for Google Sheets integration

### Deployment Steps

1. **Clone or download the project files**
2. **Ensure config.yaml and credentials.json are in the project root**
3. **Run the deployment script:**

```bash
# Make the script executable (Linux/Mac)
chmod +x deploy.sh

# Deploy the application
./deploy.sh
```

4. **Access the application at:** http://localhost:8502

## 📋 Manual Deployment

If you prefer to deploy manually:

```bash
# Build the Docker image
docker-compose build

# Start the application
docker-compose up -d

# Check status
docker-compose ps
```

## 🔧 Management Commands

```bash
# View logs
./deploy.sh logs

# Stop application
./deploy.sh stop

# Restart application
./deploy.sh restart

# Update application
./deploy.sh update

# Check status
./deploy.sh status
```

## 🌐 Port Configuration

The application runs on **port 8502** by default. To change the port:

1. **Edit docker-compose.yml:**
   ```yaml
   ports:
     - "YOUR_PORT:8502"
   ```

2. **Edit Dockerfile:**
   ```dockerfile
   ENV STREAMLIT_SERVER_PORT=8502
   EXPOSE 8502
   ```

3. **Rebuild and restart:**
   ```bash
   ./deploy.sh update
   ```

## 🔐 Environment Variables

You can set environment variables in `docker-compose.yml`:

```yaml
environment:
  - OPENAI_API_KEY=your_openai_key
  - PERPLEXITY_API_KEY=your_perplexity_key
  - GEMINI_API_KEY=your_gemini_key
```

## 📊 Monitoring

### Health Check
The application includes a health check endpoint:
- **URL:** http://localhost:8502/_stcore/health
- **Status:** Returns 200 if healthy

### Logs
```bash
# View real-time logs
docker-compose logs -f

# View recent logs
docker-compose logs --tail=100
```

## 🐛 Troubleshooting

### Port Already in Use
If port 8502 is already in use:
```bash
# Find what's using the port
lsof -i :8502

# Kill the process or change the port in docker-compose.yml
```

### Container Won't Start
```bash
# Check container logs
docker-compose logs

# Check container status
docker-compose ps

# Rebuild from scratch
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Missing Dependencies
```bash
# Rebuild with fresh dependencies
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## 🔄 Updates

To update the application:

```bash
# Pull latest changes (if using git)
git pull

# Update and restart
./deploy.sh update
```

## 📁 File Structure

```
datatobiz-brand-monitoring/
├── src/                    # Source code
├── config.yaml            # Configuration file
├── credentials.json       # Google Sheets credentials
├── streamlit_app.py       # Streamlit web application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
├── deploy.sh             # Deployment script
└── .dockerignore         # Docker ignore file
```

## 🌍 Production Deployment

For production deployment, consider:

1. **SSL/TLS**: Use a reverse proxy (nginx) with SSL certificates
2. **Authentication**: Add authentication to the Streamlit app
3. **Monitoring**: Set up monitoring and alerting
4. **Backup**: Configure automated backups for data
5. **Scaling**: Use Docker Swarm or Kubernetes for scaling

### Example with Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8502;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 📞 Support

For issues or questions:
1. Check the logs: `./deploy.sh logs`
2. Verify configuration in `config.yaml`
3. Ensure all API keys are valid
4. Check Google Sheets credentials

## 🔄 Version History

- **v2.0**: Stage 2 with multi-agent orchestration
- **v1.0**: Initial release with basic monitoring

---

**DataTobiz Brand Monitoring System** - Advanced AI-powered brand monitoring with multi-agent orchestration.

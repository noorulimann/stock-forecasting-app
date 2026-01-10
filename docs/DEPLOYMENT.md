# 🚀 Deployment Guide

This guide covers deploying the Stock Forecasting Application using Docker and Docker Compose.

---

## 📋 Prerequisites

- **Docker** 20.10+ ([Install Docker](https://docs.docker.com/get-docker/))
- **Docker Compose** 2.0+ (included with Docker Desktop)
- **4GB+ RAM** (8GB recommended)
- **10GB+ Disk Space**

---

## 🐳 Quick Start with Docker

### 1. Clone and Setup

```bash
# Clone repository
git clone <your-repo-url>
cd stock-forecasting-app

# Copy environment file
cp .env.example .env

# Edit .env with your settings (optional)
nano .env
```

### 2. Build and Start Services

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f fastapi-app
```

### 3. Access Services

- **FastAPI App**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000
- **Nginx Proxy**: http://localhost (if enabled)
- **MongoDB**: localhost:27017

### 4. Verify Deployment

```bash
# Check service status
docker-compose ps

# Test API health
curl http://localhost:8000/api/status

# Test prediction
curl -X POST http://localhost:8000/api/forecast \
  -H "Content-Type: application/json" \
  -d '{"instrument": "AAPL", "horizon": 24, "model": "LightGBM"}'
```

---

## 🔧 Configuration

### Environment Variables

Edit `.env` file for configuration:

```bash
# Application
ENVIRONMENT=production
WORKERS=4

# MongoDB
MONGO_USERNAME=admin
MONGO_PASSWORD=your-secure-password
MONGO_DATABASE=stock_data

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000

# Models
DEFAULT_MODEL=LightGBM
MODEL_SAVE_PATH=./saved_models
```

### Scaling Services

```bash
# Scale FastAPI workers
docker-compose up -d --scale fastapi-app=3

# Scale with different resources
docker-compose up -d --scale fastapi-app=5
```

---

## 📊 Service Management

### Start/Stop Services

```bash
# Start all services
docker-compose start

# Stop all services
docker-compose stop

# Restart specific service
docker-compose restart fastapi-app

# Stop and remove containers
docker-compose down

# Stop and remove containers + volumes
docker-compose down -v
```

### Update Application

```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose up -d --build

# Or rebuild specific service
docker-compose build fastapi-app
docker-compose up -d fastapi-app
```

---

## 🔍 Monitoring & Debugging

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f fastapi-app

# Last 100 lines
docker-compose logs --tail=100 fastapi-app

# Follow logs since timestamp
docker-compose logs --since 2h fastapi-app
```

### Execute Commands in Container

```bash
# Open shell in container
docker-compose exec fastapi-app /bin/sh

# Run Python command
docker-compose exec fastapi-app python -c "import lightgbm; print(lightgbm.__version__)"

# Check installed packages
docker-compose exec fastapi-app pip list
```

### Resource Usage

```bash
# View container stats
docker stats

# Inspect container
docker-compose exec fastapi-app df -h
docker-compose exec fastapi-app free -m
```

---

## 🗄️ Data Persistence

### Volumes

Docker Compose creates named volumes for persistence:

- `stock-mongodb-data` - MongoDB database
- `stock-mlflow-data` - MLflow experiments
- `./saved_models` - Trained models (host mount)
- `./logs` - Application logs (host mount)

### Backup Data

```bash
# Backup MongoDB
docker-compose exec mongodb mongodump --out /data/backup

# Backup MLflow data
docker cp stock-mlflow:/mlflow ./mlflow_backup

# Backup saved models
tar -czf saved_models_backup.tar.gz saved_models/
```

### Restore Data

```bash
# Restore MongoDB
docker-compose exec mongodb mongorestore /data/backup

# Restore MLflow data
docker cp ./mlflow_backup stock-mlflow:/mlflow
```

---

## 🔒 Security Best Practices

### 1. Change Default Passwords

```bash
# Edit .env
MONGO_PASSWORD=<strong-random-password>
SECRET_KEY=<generate-with-openssl-rand-base64-32>
```

### 2. Enable SSL/TLS (Production)

```bash
# Generate SSL certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/nginx.key \
  -out nginx/ssl/nginx.crt

# Update nginx.conf for HTTPS
# See nginx/ssl-example.conf
```

### 3. Restrict Network Access

```bash
# Only expose necessary ports
# Edit docker-compose.yml:
ports:
  - "127.0.0.1:8000:8000"  # Only localhost
```

### 4. Use Docker Secrets (Production)

```yaml
# docker-compose.prod.yml
secrets:
  mongo_password:
    file: ./secrets/mongo_password.txt
    
services:
  mongodb:
    secrets:
      - mongo_password
```

---

## 📈 Performance Optimization

### 1. Increase Worker Processes

```bash
# .env
WORKERS=4  # Set based on CPU cores
```

### 2. Enable Caching

```bash
# .env
ENABLE_MODEL_CACHE=true
CACHE_TTL=300
```

### 3. Optimize Docker Build

```bash
# Use BuildKit for faster builds
DOCKER_BUILDKIT=1 docker-compose build

# Multi-platform builds
docker buildx build --platform linux/amd64,linux/arm64 .
```

### 4. Resource Limits

```yaml
# docker-compose.yml
services:
  fastapi-app:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

---

## 🌍 Production Deployment

### AWS EC2 Deployment

```bash
# 1. Launch EC2 instance (Ubuntu 22.04, t3.medium+)
# 2. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# 3. Clone and deploy
git clone <your-repo>
cd stock-forecasting-app
cp .env.example .env
# Edit .env with production values
docker-compose up -d

# 4. Setup domain and SSL (optional)
# Use Let's Encrypt with certbot
```

### Azure Container Instances

```bash
# 1. Build and push to Azure Container Registry
az acr build --registry <your-acr> --image stock-api:latest .

# 2. Deploy to ACI
az container create \
  --resource-group <rg-name> \
  --name stock-api \
  --image <your-acr>.azurecr.io/stock-api:latest \
  --cpu 2 --memory 4 \
  --ports 8000 \
  --environment-variables MONGODB_URI=<connection-string>
```

### Kubernetes Deployment (Advanced)

```bash
# 1. Create Kubernetes manifests
kubectl apply -f k8s/

# 2. Expose service
kubectl expose deployment stock-api --type=LoadBalancer --port=8000

# 3. Scale deployment
kubectl scale deployment stock-api --replicas=5
```

---

## 🧪 Testing Deployment

### Health Checks

```bash
# API health
curl http://localhost:8000/api/status

# MLflow health
curl http://localhost:5000/health

# MongoDB health
docker-compose exec mongodb mongosh --eval "db.adminCommand('ping')"
```

### Load Testing

```bash
# Install Apache Bench
sudo apt-get install apache2-utils

# Test API endpoint
ab -n 1000 -c 10 http://localhost:8000/api/status

# Test prediction endpoint
ab -n 100 -c 5 -p request.json -T application/json \
  http://localhost:8000/api/forecast
```

---

## ❓ Troubleshooting

### Common Issues

**Port Already in Use:**
```bash
# Find process using port
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows

# Kill process or change port in docker-compose.yml
```

**Container Crashes:**
```bash
# View crash logs
docker-compose logs --tail=50 fastapi-app

# Check container status
docker-compose ps

# Inspect container
docker inspect stock-fastapi
```

**MongoDB Connection Failed:**
```bash
# Check MongoDB is running
docker-compose ps mongodb

# Test connection
docker-compose exec mongodb mongosh

# Check credentials in .env
```

**Out of Memory:**
```bash
# Increase Docker memory limit
# Docker Desktop > Settings > Resources > Memory

# Or reduce workers
# .env: WORKERS=1
```

---

## 📞 Support

For issues or questions:
- GitHub Issues: [Create Issue](https://github.com/yourusername/stock-forecasting-app/issues)
- Documentation: [docs/](./docs/)
- Email: your.email@example.com

---

**Last Updated:** January 2026

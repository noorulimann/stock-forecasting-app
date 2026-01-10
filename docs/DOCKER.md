# 🐳 Docker Quick Reference Guide

## Quick Start Commands

### Windows (PowerShell)

```powershell
# Quick start with script
.\start.bat

# Manual start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose stop
```

### Linux/Mac (Bash)

```bash
# Quick start with script
chmod +x start.sh
./start.sh

# Manual start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose stop
```

---

## Service URLs

| Service | URL | Description |
|---------|-----|-------------|
| FastAPI App | http://localhost:8000 | Main web interface |
| API Docs (Swagger) | http://localhost:8000/docs | Interactive API documentation |
| API Docs (ReDoc) | http://localhost:8000/redoc | Alternative API docs |
| MLflow UI | http://localhost:5000 | Experiment tracking |
| MongoDB | localhost:27017 | Database (internal) |
| Nginx | http://localhost | Reverse proxy (optional) |

---

## Common Commands

### Service Management

```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d fastapi-app

# Stop all services
docker-compose stop

# Restart service
docker-compose restart fastapi-app

# Stop and remove containers
docker-compose down

# Stop and remove containers + volumes (⚠️ deletes data)
docker-compose down -v
```

### Logs & Monitoring

```bash
# View all logs (follow)
docker-compose logs -f

# View specific service logs
docker-compose logs -f fastapi-app

# View last 100 lines
docker-compose logs --tail=100 fastapi-app

# View logs since 2 hours ago
docker-compose logs --since 2h

# Check container status
docker-compose ps

# View resource usage
docker stats
```

### Development Mode

```bash
# Start with hot reload
docker-compose -f docker-compose.dev.yml up -d

# View dev logs
docker-compose -f docker-compose.dev.yml logs -f

# Stop dev environment
docker-compose -f docker-compose.dev.yml down
```

### Scaling

```bash
# Scale FastAPI workers
docker-compose up -d --scale fastapi-app=3

# Check scaled instances
docker-compose ps fastapi-app
```

### Execute Commands in Container

```bash
# Open shell in container
docker-compose exec fastapi-app /bin/sh

# Run Python command
docker-compose exec fastapi-app python -c "import lightgbm; print(lightgbm.__version__)"

# Check installed packages
docker-compose exec fastapi-app pip list

# Run management command
docker-compose exec fastapi-app python -m pytest
```

### Database Operations

```bash
# MongoDB shell
docker-compose exec mongodb mongosh

# MongoDB admin shell
docker-compose exec mongodb mongosh -u admin -p stockforecast2026

# Backup database
docker-compose exec mongodb mongodump --out /data/backup

# Restore database
docker-compose exec mongodb mongorestore /data/backup

# Check database stats
docker-compose exec mongodb mongosh --eval "db.stats()"
```

---

## Testing

### API Health Check

```bash
# Test API status
curl http://localhost:8000/api/status

# Test prediction
curl -X POST http://localhost:8000/api/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "instrument": "AAPL",
    "horizon": 24,
    "model": "LightGBM"
  }'
```

### Service Health Checks

```bash
# Check all services
docker-compose ps

# Check specific health
docker inspect --format='{{.State.Health.Status}}' stock-fastapi
docker inspect --format='{{.State.Health.Status}}' stock-mongodb
docker inspect --format='{{.State.Health.Status}}' stock-mlflow
```

---

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 8000
# Linux/Mac
lsof -i :8000

# Windows (PowerShell)
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess

# Change port in docker-compose.yml
ports:
  - "8001:8000"  # Use 8001 instead
```

### Container Crashes

```bash
# View crash logs
docker-compose logs --tail=50 fastapi-app

# Inspect container
docker inspect stock-fastapi

# Check container details
docker-compose ps
```

### Reset Everything

```bash
# ⚠️ Nuclear option - removes everything
docker-compose down -v
docker system prune -a --volumes

# Then rebuild
docker-compose up -d --build
```

### Out of Memory

```bash
# Increase Docker memory limit
# Docker Desktop > Settings > Resources > Memory (set to 4GB+)

# Or reduce workers in .env
WORKERS=1
```

### MongoDB Connection Issues

```bash
# Check MongoDB is running
docker-compose ps mongodb

# Test connection
docker-compose exec mongodb mongosh --eval "db.adminCommand('ping')"

# Check logs
docker-compose logs mongodb

# Verify credentials in .env
cat .env | grep MONGO
```

---

## Build & Optimization

### Rebuild Containers

```bash
# Rebuild all
docker-compose build

# Rebuild specific service
docker-compose build fastapi-app

# Build without cache
docker-compose build --no-cache

# Build and start
docker-compose up -d --build
```

### Image Management

```bash
# List images
docker images

# Remove unused images
docker image prune

# Remove specific image
docker rmi stock-forecasting-app_fastapi-app

# Check image size
docker images | grep stock
```

### Volume Management

```bash
# List volumes
docker volume ls

# Inspect volume
docker volume inspect stock-mongodb-data

# Backup volume
docker run --rm -v stock-mongodb-data:/data -v $(pwd):/backup \
  alpine tar czf /backup/mongodb_backup.tar.gz /data

# Remove unused volumes
docker volume prune
```

---

## Production Checklist

- [ ] Update `.env` with secure passwords
- [ ] Set `ENVIRONMENT=production`
- [ ] Configure `WORKERS` based on CPU cores
- [ ] Enable SSL/TLS (nginx configuration)
- [ ] Set up monitoring and alerts
- [ ] Configure backups (MongoDB, models)
- [ ] Test health checks
- [ ] Review resource limits in docker-compose.yml
- [ ] Set up log rotation
- [ ] Configure firewall rules

---

## Performance Tips

1. **Use BuildKit for faster builds**
   ```bash
   DOCKER_BUILDKIT=1 docker-compose build
   ```

2. **Enable Docker layer caching**
   - Order Dockerfile commands from least to most frequently changed

3. **Optimize worker count**
   ```bash
   WORKERS=<number_of_cpu_cores>
   ```

4. **Use production WSGI server**
   - Already configured: uvicorn with multiple workers

5. **Enable model caching**
   ```bash
   ENABLE_MODEL_CACHE=true
   ```

---

## Support

- **Documentation**: [docs/DEPLOYMENT.md](../docs/DEPLOYMENT.md)
- **Issues**: Create GitHub issue
- **Logs**: `docker-compose logs -f`

---

**Last Updated**: January 2026

@echo off
REM ============================================================================
REM Stock Forecasting App - Quick Start Script (Windows)
REM ============================================================================

echo.
echo 🚀 Stock Forecasting Application - Docker Setup
echo ==============================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not installed. Please install Docker Desktop first.
    echo    Visit: https://docs.docker.com/desktop/install/windows-install/
    pause
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if errorlevel 1 (
    docker compose version >nul 2>&1
    if errorlevel 1 (
        echo ❌ Docker Compose is not installed.
        pause
        exit /b 1
    )
)

echo ✅ Docker installed
echo ✅ Docker Compose installed
echo.

REM Create .env if it doesn't exist
if not exist .env (
    echo 📝 Creating .env file from .env.example...
    copy .env.example .env >nul
    echo ✅ .env created (you may want to customize it^)
) else (
    echo ✅ .env file already exists
)
echo.

REM Create required directories
echo 📁 Creating required directories...
if not exist saved_models mkdir saved_models
if not exist mlruns mkdir mlruns
if not exist logs mkdir logs
if not exist nginx\ssl mkdir nginx\ssl
echo ✅ Directories created
echo.

REM Menu
echo Choose an option:
echo 1^) Start all services (Production^)
echo 2^) Start all services (Development with hot reload^)
echo 3^) Build only (no start^)
echo 4^) Stop all services
echo 5^) Stop and remove all (including volumes^)
echo.
set /p choice="Enter choice [1-5]: "

if "%choice%"=="1" goto prod
if "%choice%"=="2" goto dev
if "%choice%"=="3" goto build
if "%choice%"=="4" goto stop
if "%choice%"=="5" goto remove
goto invalid

:prod
echo.
echo 🚀 Starting all services in production mode...
docker-compose up -d
goto status

:dev
echo.
echo 🚀 Starting all services in development mode...
docker-compose -f docker-compose.dev.yml up -d
goto status

:build
echo.
echo 🔨 Building images...
docker-compose build
goto end

:stop
echo.
echo ⏹️ Stopping services...
docker-compose stop
docker-compose -f docker-compose.dev.yml stop 2>nul
goto end

:remove
echo.
echo ⚠️  This will remove all containers and volumes!
set /p confirm="Are you sure? [y/N]: "
if /i not "%confirm%"=="y" (
    echo ❌ Cancelled
    goto end
)
docker-compose down -v
docker-compose -f docker-compose.dev.yml down -v 2>nul
echo ✅ All services and volumes removed
goto end

:invalid
echo ❌ Invalid choice
goto end

:status
REM Wait for services
echo.
echo ⏳ Waiting for services to be ready...
timeout /t 10 /nobreak >nul

REM Check status
echo.
echo 📊 Service Status:
docker-compose ps

echo.
echo ✅ Setup complete!
echo.
echo 🌐 Access your services:
echo    - FastAPI App:      http://localhost:8000
echo    - API Docs:         http://localhost:8000/docs
echo    - MLflow UI:        http://localhost:5000
echo    - MongoDB:          localhost:27017
echo.
echo 📝 Useful commands:
echo    View logs:          docker-compose logs -f
echo    Stop services:      docker-compose stop
echo    Restart:            docker-compose restart
echo.
echo 📖 For more info, see docs\DEPLOYMENT.md

:end
echo.
pause

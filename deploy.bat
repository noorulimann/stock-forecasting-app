@echo off
REM Automated Deployment Script for Vercel (Windows)
REM Stock Forecasting App - Phase 1

echo ==================================================
echo   STOCK FORECASTING APP - VERCEL DEPLOYMENT
echo ==================================================
echo.

REM Step 1: Check if git is initialized
echo Step 1: Checking Git setup...
if exist ".git" (
    echo [32m✓ Git repository found[0m
) else (
    echo [33m! Git not initialized. Initializing...[0m
    git init
    git branch -M main
)

REM Step 2: Check remote
echo.
echo Step 2: Setting up remote repository...
git remote remove origin 2>nul
git remote add origin https://github.com/noorulimann/stock-forecasting-app-vercel.git
echo [32m✓ Remote set to: https://github.com/noorulimann/stock-forecasting-app-vercel[0m

REM Step 3: Add all files
echo.
echo Step 3: Staging files for commit...
git add .
echo [32m✓ All files staged[0m

REM Step 4: Commit
echo.
echo Step 4: Creating commit...
git commit -m "Deploy Phase 1: LightGBM End-to-End ML Pipeline - Added LightGBM model with hyperparameter optimization - Implemented feature engineering with Pandas - Added model comparison and visualization tools - Created REST API endpoints for ML serving - Configured for Vercel serverless deployment - Ready for production with MongoDB Atlas"
echo [32m✓ Commit created[0m

REM Step 5: Push to GitHub
echo.
echo Step 5: Pushing to GitHub...
echo [33m! You may need to enter your GitHub credentials[0m
git push -u origin main --force

echo.
echo ==================================================
echo   DEPLOYMENT PREPARATION COMPLETE!
echo ==================================================
echo.
echo Next steps:
echo 1. Go to: https://vercel.com/new
echo 2. Click 'Import Git Repository'
echo 3. Select: noorulimann/stock-forecasting-app-vercel
echo 4. Configure environment variables (see VERCEL_SETUP.txt)
echo 5. Click 'Deploy'
echo.
echo ==================================================
pause

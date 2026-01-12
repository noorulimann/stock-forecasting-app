#!/bin/bash
# Automated Deployment Script for Vercel
# Stock Forecasting App - Phase 1

echo "=================================================="
echo "  STOCK FORECASTING APP - VERCEL DEPLOYMENT"
echo "=================================================="
echo ""

# Step 1: Check if git is initialized
echo "Step 1: Checking Git setup..."
if [ -d ".git" ]; then
    echo "✅ Git repository found"
else
    echo "❌ Git not initialized"
    echo "Initializing git..."
    git init
    git branch -M main
fi

# Step 2: Check remote
echo ""
echo "Step 2: Setting up remote repository..."
git remote remove origin 2>/dev/null
git remote add origin https://github.com/noorulimann/stock-forecasting-app-vercel.git
echo "✅ Remote set to: https://github.com/noorulimann/stock-forecasting-app-vercel"

# Step 3: Add all files
echo ""
echo "Step 3: Staging files for commit..."
git add .
echo "✅ All files staged"

# Step 4: Commit
echo ""
echo "Step 4: Creating commit..."
git commit -m "Deploy Phase 1: LightGBM End-to-End ML Pipeline

- Added LightGBM model with hyperparameter optimization
- Implemented feature engineering with Pandas
- Added model comparison and visualization tools
- Created REST API endpoints for ML serving
- Configured for Vercel serverless deployment
- Ready for production with MongoDB Atlas"

echo "✅ Commit created"

# Step 5: Push to GitHub
echo ""
echo "Step 5: Pushing to GitHub..."
echo "⚠️  You may need to enter your GitHub credentials"
git push -u origin main --force

echo ""
echo "=================================================="
echo "  DEPLOYMENT PREPARATION COMPLETE!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Go to: https://vercel.com/new"
echo "2. Click 'Import Git Repository'"
echo "3. Select: noorulimann/stock-forecasting-app-vercel"
echo "4. Configure environment variables (see below)"
echo "5. Click 'Deploy'"
echo ""
echo "Environment Variables to add in Vercel:"
echo "----------------------------------------"
echo "MONGODB_URI = mongodb+srv://forecasting_user:noor123iman@cluster0.fdma2vg.mongodb.net/forecasting_db?retryWrites=true&w=majority&appName=Cluster0"
echo "SECRET_KEY = $(openssl rand -hex 32 2>/dev/null || echo 'generate-a-secret-key-here')"
echo "DATABASE_NAME = forecasting_db"
echo "PYTHON_VERSION = 3.9"
echo ""
echo "=================================================="

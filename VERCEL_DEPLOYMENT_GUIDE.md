# 🚀 Vercel Deployment Guide

## Complete Configuration for Stock Forecasting App

---

## 📋 Step-by-Step Deployment

### **Step 1: Vercel Project Settings**

In your Vercel dashboard (https://vercel.com/dashboard), go to your project settings:

#### **A. Build and Deployment Section**

| Setting | Value |
|---------|-------|
| **Framework Preset** | `Other` |
| **Build Command** | `pip install -r requirements-vercel.txt` |
| **Output Directory** | `.` (leave as default) |
| **Install Command** | `pip install -r requirements-vercel.txt` |
| **Development Command** | `python app.py` |
| **Root Directory** | `./` (leave blank or use `./`) |

#### **B. Environment Variables Section**

Click "Environment Variables" and add these one by one:

##### **Required Variables:**

1. **MONGODB_URI**
   - **Value:** Your MongoDB connection string
   - **Example:** `mongodb+srv://username:password@cluster.mongodb.net/forecasting_db`
   - **Environments:** ✅ Production, ✅ Preview, ✅ Development
   - **How to get:**
     - Go to MongoDB Atlas (https://cloud.mongodb.com)
     - Click "Connect" → "Connect your application"
     - Copy the connection string
     - Replace `<password>` with your actual password

2. **SECRET_KEY**
   - **Value:** A random secret key for Flask sessions
   - **Example:** `your-super-secret-key-change-this-in-production`
   - **Environments:** ✅ Production, ✅ Preview, ✅ Development
   - **Generate one:**
     ```python
     import secrets
     print(secrets.token_hex(32))
     ```

3. **DATABASE_NAME**
   - **Value:** `forecasting_db`
   - **Environments:** ✅ Production, ✅ Preview, ✅ Development

4. **PYTHON_VERSION**
   - **Value:** `3.9`
   - **Environments:** ✅ Production, ✅ Preview, ✅ Development

##### **Optional Variables (for Phase 1 features):**

5. **DEFAULT_INSTRUMENT**
   - **Value:** `AAPL`
   - **Environments:** ✅ Production, ✅ Preview, ✅ Development

6. **MODEL_SAVE_PATH**
   - **Value:** `/tmp/saved_models/`
   - **Environments:** ✅ Production, ✅ Preview, ✅ Development
   - **Note:** Vercel uses `/tmp` for temporary storage

---

### **Step 2: MongoDB Setup (Required)**

Since Vercel is serverless, you need a cloud MongoDB instance:

1. **Go to MongoDB Atlas:** https://www.mongodb.com/cloud/atlas
2. **Create a free cluster** (M0 Free Tier)
3. **Create Database User:**
   - Username: `your-username`
   - Password: `your-password` (save this!)
4. **Whitelist IP Addresses:**
   - Click "Network Access"
   - Click "Add IP Address"
   - Select "Allow Access from Anywhere" (`0.0.0.0/0`)
   - Click "Confirm"
5. **Get Connection String:**
   - Click "Connect" → "Connect your application"
   - Copy the connection string
   - Replace `<password>` with your actual password
   - Paste in Vercel Environment Variables as `MONGODB_URI`

---

### **Step 3: Git Configuration**

Your Git repository is already connected! ✅

**What happens on deployment:**
- Every push to `main` branch → Automatic deployment to Production
- Every push to other branches → Preview deployment
- Pull requests → Automatic preview deployments

---

### **Step 4: Deploy**

#### **Option A: Deploy from Git (Automatic)**

1. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Configure for Vercel deployment"
   git push origin main
   ```

2. **Vercel automatically deploys** when you push to GitHub

3. **Check deployment status:**
   - Go to Vercel Dashboard → Deployments
   - Watch the build logs
   - Wait for "Ready" status

#### **Option B: Deploy from Vercel Dashboard (Manual)**

1. Go to your project in Vercel
2. Click "Deployments" tab
3. Click "Redeploy" on the latest deployment
4. Select "Use existing Build Cache" or "Rebuild"
5. Click "Redeploy"

---

### **Step 5: Verify Deployment**

Once deployed, you'll get a URL like:
```
https://stock-forecasting-app-vercel.vercel.app
```

**Test these endpoints:**

1. **Homepage:**
   ```
   https://your-app.vercel.app/
   ```

2. **API Health Check:**
   ```
   https://your-app.vercel.app/api/instruments
   ```

3. **LightGBM Training (Phase 1):**
   ```bash
   curl -X POST https://your-app.vercel.app/api/lightgbm/train/AAPL \
     -H "Content-Type: application/json" \
     -d '{"optimize": true}'
   ```

---

## 🔧 Configuration Files

### **vercel.json** (Already created ✅)

Located at: `stock-forecasting-app/vercel.json`

```json
{
  "version": 2,
  "builds": [
    {
      "src": "app.py",
      "use": "@vercel/python"
    },
    {
      "src": "static/**",
      "use": "@vercel/static"
    }
  ],
  "routes": [
    {
      "src": "/static/(.*)",
      "dest": "/static/$1"
    },
    {
      "src": "/(.*)",
      "dest": "app.py"
    }
  ],
  "env": {
    "PYTHON_VERSION": "3.9"
  }
}
```

### **requirements-vercel.txt** (Already created ✅)

Located at: `stock-forecasting-app/requirements-vercel.txt`

Contains lightweight dependencies optimized for serverless deployment.

---

## ⚠️ Important Limitations on Vercel

### **1. Serverless Function Size**
- **Limit:** 50MB compressed, 250MB uncompressed
- **Solution:** Use `requirements-vercel.txt` (minimal dependencies)
- **Excluded:** Heavy ML libraries (PyTorch, TensorFlow) - Use LightGBM instead ✅

### **2. Execution Time**
- **Hobby Plan:** 10 seconds max
- **Pro Plan:** 60 seconds max
- **Solution:** Optimize model training (LightGBM is fast ✅)

### **3. File System**
- **Read-only** except `/tmp` directory
- **Solution:** Save models to `/tmp` or cloud storage
- **Note:** `/tmp` is cleared between invocations

### **4. Memory**
- **Hobby Plan:** 1024 MB
- **Pro Plan:** 3008 MB
- **Solution:** LightGBM is memory-efficient ✅

---

## 🛠️ Troubleshooting

### **Problem: Build Fails**

**Solution:**
```bash
# Check build logs in Vercel
# Common issues:
1. Missing dependencies → Add to requirements-vercel.txt
2. Import errors → Check Python version (use 3.9)
3. Large dependencies → Remove heavy packages
```

### **Problem: MongoDB Connection Error**

**Solution:**
```bash
# Check:
1. MONGODB_URI environment variable is set correctly
2. Password is correct (no special characters causing issues)
3. IP whitelist includes 0.0.0.0/0
4. Database user has correct permissions
```

### **Problem: 500 Internal Server Error**

**Solution:**
```bash
# Check Vercel function logs:
1. Go to Deployments → Click deployment → View Function Logs
2. Look for Python errors
3. Check if all environment variables are set
```

### **Problem: Model Training Times Out**

**Solution:**
```python
# Reduce model complexity:
forecaster.fit(
    data=data,
    optimize=False,  # Disable optimization for faster training
    num_boost_round=100  # Reduce from 500
)
```

---

## 🎯 Recommended Workflow

### **Development:**
```bash
# Work locally
python app.py

# Test locally on http://localhost:5000
```

### **Staging:**
```bash
# Push to a feature branch
git checkout -b feature/new-feature
git push origin feature/new-feature

# Vercel creates preview deployment
# Test at: https://stock-forecasting-app-git-feature-*.vercel.app
```

### **Production:**
```bash
# Merge to main
git checkout main
git merge feature/new-feature
git push origin main

# Vercel deploys to production
# Live at: https://stock-forecasting-app-vercel.vercel.app
```

---

## 📊 Monitoring

### **View Logs:**
1. Go to Vercel Dashboard
2. Click your project
3. Click "Logs" tab
4. Filter by:
   - All logs
   - Errors only
   - Specific functions

### **Analytics:**
1. Click "Analytics" tab
2. View:
   - Request count
   - Response times
   - Error rates
   - Geographic distribution

---

## 🚀 Performance Optimization

### **1. Use Environment Variables**
```python
import os
MONGODB_URI = os.environ.get('MONGODB_URI')
```

### **2. Implement Caching**
```python
# Cache model predictions
from functools import lru_cache

@lru_cache(maxsize=128)
def get_prediction(symbol):
    # Your prediction logic
    pass
```

### **3. Lazy Loading**
```python
# Import heavy libraries only when needed
def train_model():
    from models.lightgbm_model import get_lightgbm_forecaster
    forecaster = get_lightgbm_forecaster()
    # Training logic
```

---

## 📱 Custom Domain (Optional)

### **Add Your Domain:**

1. Go to Project Settings → Domains
2. Click "Add Domain"
3. Enter your domain: `stockforecast.yourdomain.com`
4. Add DNS records:
   - **Type:** CNAME
   - **Name:** stockforecast (or www)
   - **Value:** cname.vercel-dns.com
5. Wait for DNS propagation (up to 48 hours)

---

## ✅ Deployment Checklist

Before deploying to production:

- [ ] MongoDB Atlas cluster created and configured
- [ ] All environment variables set in Vercel
- [ ] IP whitelist configured (0.0.0.0/0)
- [ ] `vercel.json` file present in root
- [ ] `requirements-vercel.txt` optimized
- [ ] Git repository connected
- [ ] Test deployment in preview environment
- [ ] API endpoints tested
- [ ] Error handling implemented
- [ ] Logging configured

---

## 🎉 Success!

Once deployed, your application will be live at:
```
https://stock-forecasting-app-vercel.vercel.app
```

**Features Available:**
- ✅ Real-time stock data collection
- ✅ LightGBM model training (Phase 1)
- ✅ 7-day forecasts
- ✅ Feature importance analysis
- ✅ Model comparison
- ✅ REST API endpoints
- ✅ Automatic deployments on git push

---

## 📚 Additional Resources

- **Vercel Docs:** https://vercel.com/docs
- **Vercel Python:** https://vercel.com/docs/concepts/functions/serverless-functions/runtimes/python
- **MongoDB Atlas:** https://docs.atlas.mongodb.com/
- **Project README:** See `README.md` for full documentation

---

**Last Updated:** January 12, 2026
**Status:** Ready for Deployment ✅

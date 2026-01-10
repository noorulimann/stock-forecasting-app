# 📈 Stock Forecasting Application

> A production-ready machine learning pipeline for financial forecasting with ensemble methods, modern gradient boosting, and deep learning models.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Latest-yellow.svg)](https://lightgbm.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)](https://mlflow.org/)

---

## 🎯 Overview

A comprehensive end-to-end machine learning pipeline for stock price forecasting, demonstrating industry-standard practices in data engineering, model training, experiment tracking, and deployment.

### Key Features

- **Multiple ML Models**: LightGBM, LSTM, GRU, ARIMA, and ensemble methods
- **Modern API**: FastAPI with async endpoints and automatic OpenAPI documentation
- **Experiment Tracking**: MLflow integration for model versioning and metrics
- **Production Ready**: Docker containerization with health checks and monitoring
- **Real-time Data**: Automated collection from Yahoo Finance with technical indicators
- **Interactive UI**: Professional web interface for forecasting and analysis

---

## 🏗️ Architecture

```
┌─────────────────┐
│  Data Sources   │  Yahoo Finance API
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Pipeline   │  Collection, Cleaning, Feature Engineering
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│       ML Models                     │
│  • LightGBM (Gradient Boosting)     │
│  • LSTM & GRU (Deep Learning)       │
│  • ARIMA (Time Series)              │
│  • Ensemble Methods (5 strategies)  │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  MLflow Server  │  Experiment Tracking & Model Registry
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FastAPI Server │  REST API for Predictions
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Docker Deploy  │  Containerized Production Environment
└─────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker (optional, for containerized deployment)
- MongoDB (optional, for data persistence)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd stock-forecasting-app
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   - Web Interface: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - MLflow UI: http://localhost:5000

---

## 🐳 Docker Deployment

### Development Mode
```bash
docker-compose -f docker-compose.dev.yml up
```

### Production Mode
```bash
docker-compose up -d
```

The application will be available at:
- Application: http://localhost:8000
- MLflow: http://localhost:5000
- MongoDB: localhost:27017

---

## 📊 Supported Instruments

### Stocks
- **AAPL** - Apple Inc.
- **GOOGL** - Alphabet Inc.
- **MSFT** - Microsoft Corp.
- **TSLA** - Tesla Inc.

### Cryptocurrencies
- **BTC-USD** - Bitcoin
- **ETH-USD** - Ethereum

---

## 🤖 Models

### 1. LightGBM
- Modern gradient boosting framework
- Automated hyperparameter optimization
- Feature importance analysis
- Best for: Fast training and high accuracy

### 2. Neural Networks
- **LSTM**: Long Short-Term Memory
- **GRU**: Gated Recurrent Unit
- Best for: Capturing complex temporal patterns

### 3. Traditional Models
- **ARIMA**: AutoRegressive Integrated Moving Average
- **SMA**: Simple Moving Average
- **EMA**: Exponential Moving Average
- Best for: Baseline comparisons

### 4. Ensemble Methods
- Simple Average
- Weighted Average
- Performance Weighted
- Dynamic Selection
- Rank-Based (Borda Count)
- Best for: Robust predictions with reduced variance

---

## 🛠️ API Endpoints

### Forecasting
```http
POST /api/forecast
Content-Type: application/json

{
  "instrument": "AAPL",
  "horizon": 24,
  "model": "LightGBM"
}
```

### Ensemble Analysis
```http
POST /api/ensemble/predict
Content-Type: application/json

{
  "instrument": "AAPL",
  "strategy": "performance_weighted",
  "horizon": 24
}
```

### Model Training
```http
POST /api/v1/lightgbm/train/AAPL
Content-Type: application/json

{
  "optimize": true,
  "n_trials": 50
}
```

### System Status
```http
GET /api/status
```

For complete API documentation, visit `/docs` after starting the server.

---

## 📁 Project Structure

```
stock-forecasting-app/
├── data/                   # Data collection and processing
│   ├── collector.py       # Yahoo Finance data fetching
│   ├── database.py        # MongoDB integration
│   └── processor.py       # Feature engineering
├── models/                 # ML model implementations
│   ├── lightgbm_model.py  # LightGBM forecaster
│   ├── neural.py          # LSTM/GRU models
│   ├── traditional.py     # ARIMA, SMA, EMA
│   ├── ensemble.py        # Ensemble methods
│   └── trainer.py         # Training pipeline
├── utils/                  # Utility functions
│   ├── evaluator.py       # Model evaluation metrics
│   ├── lightgbm_viz.py    # Visualization tools
│   └── model_comparison.py # Model benchmarking
├── templates/              # HTML templates
│   ├── index.html         # Dashboard
│   ├── forecast.html      # Forecasting interface
│   ├── ensemble.html      # Ensemble analysis
│   └── performance.html   # Model performance
├── static/                 # Static assets
│   ├── css/               # Stylesheets
│   └── js/                # JavaScript
├── tests/                  # Test suite
├── docs/                   # Documentation
├── app.py                 # Main FastAPI application
├── config.py              # Configuration
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker image definition
└── docker-compose.yml     # Docker orchestration
```

---

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run specific test modules:
```bash
pytest tests/test_api_integration.py
pytest tests/test_data_pipeline.py
pytest tests/test_end_to_end.py
```

---

## 📈 Performance Metrics

The application tracks the following metrics:

- **MAPE**: Mean Absolute Percentage Error
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination
- **Directional Accuracy**: Percentage of correct trend predictions

All metrics are logged to MLflow for experiment tracking and comparison.

---

## 🔧 Configuration

Edit `.env` file to configure:

```env
# Application
APP_HOST=0.0.0.0
APP_PORT=8000

# MongoDB
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DB=stock_forecasting

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# API Keys
ALPHA_VANTAGE_KEY=your_key_here  # Optional
```

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions or support, please open an issue in the repository.

---

## 🙏 Acknowledgments

- **Yahoo Finance** for providing financial data
- **LightGBM** team for the excellent gradient boosting framework
- **FastAPI** team for the modern web framework
- **MLflow** for experiment tracking capabilities

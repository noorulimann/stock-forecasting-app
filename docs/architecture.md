# Stock Forecasting Application - Architecture Documentation

## 📊 System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           STOCK FORECASTING APPLICATION                             │
│                              Flask Web Framework                                    │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                 PRESENTATION LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  Frontend Components:                                                               │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐              │
│  │  index.html  │ │forecast.html │ │ensemble.html │ │performance.  │              │
│  │   (Main UI)  │ │ (Forecasts)  │ │(Ensemble Res)│ │   html       │              │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘              │
│                                                                                     │
│  Interactive Elements:                                                              │
│  • Instrument Selection (Stocks/Crypto/Forex)                                      │
│  • Forecast Horizon Selection (1hr, 3hrs, 24hrs, 72hrs)                          │
│  • Model Selection Interface                                                       │
│  • Real-time Plotly Charts with Candlestick + Forecast Overlay                    │
│                                                                                     │
│  JavaScript/CSS:                                                                   │
│  • Bootstrap 5 for responsive design                                               │
│  • Plotly.js for interactive charts                                                │
│  • Custom CSS animations                                                           │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                   HTTP/JSON API
                                        │
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                               APPLICATION LAYER (Flask)                            │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  Main Application (app.py):                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  Web Routes:           │  API Routes:                                       │   │
│  │  • / (Homepage)        │  • /api/forecast                                   │   │
│  │  • /forecast           │  • /api/performance                                │   │
│  │  • /ensemble           │  • /api/ensemble/predict                           │   │
│  │  • /performance        │  • /api/ensemble/compare                           │   │
│  │                        │  • /api/instruments                                │   │
│  │                        │  • /api/status                                     │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│  Route Handlers (routes.py):                                                       │
│  • Request validation and preprocessing                                            │
│  • Model orchestration and prediction calls                                       │
│  • Response formatting and error handling                                         │
│  • Performance metrics calculation                                                │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                               BUSINESS LOGIC LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  Model Layer (models/):                                                            │
│                                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                    │
│  │  Traditional    │  │     Neural      │  │    Ensemble     │                    │
│  │    Models       │  │    Models       │  │     Models      │                    │
│  │  ┌───────────┐  │  │  ┌───────────┐  │  │  ┌───────────┐  │                    │
│  │  │    SMA    │  │  │  │   LSTM    │  │  │  │Performance│  │                    │
│  │  │    EMA    │  │  │  │    GRU    │  │  │  │ Weighted  │  │                    │
│  │  │   ARIMA   │  │  │  │Transformer│  │  │  │  Simple   │  │                    │
│  │  │    VAR    │  │  │  │  (Ready)  │  │  │  │ Average   │  │                    │
│  │  └───────────┘  │  │  └───────────┘  │  │  │  Dynamic  │  │                    │
│  └─────────────────┘  └─────────────────┘  │  │ Selection │  │                    │
│                                            │  │Rank Based │  │                    │
│  Model Training & Management:              │  └───────────┘  │                    │
│  ┌─────────────────────────────────────┐   └─────────────────┘                    │
│  │         trainer.py                  │                                          │
│  │  • Unified training pipeline       │   Model Utilities:                       │
│  │  • Model performance evaluation    │   ┌─────────────────────────────────────┐ │
│  │  • Cross-validation framework      │   │         evaluator.py               │ │
│  │  • Hyperparameter optimization     │   │  • RMSE, MAE, MAPE calculation     │ │
│  │  • Model comparison utilities      │   │  • R² and directional accuracy     │ │
│  └─────────────────────────────────────┘   │  • Model ranking and comparison    │ │
│                                            │  • Performance visualization       │ │
│                                            └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                DATA ACCESS LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  Data Management (data/):                                                          │
│                                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                    │
│  │    Collector    │  │    Processor    │  │    Database     │                    │
│  │   (collector.py)│  │  (processor.py) │  │  (database.py)  │                    │
│  │                 │  │                 │  │                 │                    │
│  │ • yfinance API  │  │ • Technical     │  │ • MongoDB Ops   │                    │
│  │ • Data cleaning │  │   Indicators    │  │ • Collection    │                    │
│  │ • Multi-asset   │  │ • Feature Eng   │  │   Management    │                    │
│  │   support       │  │ • Normalization │  │ • Index Setup   │                    │
│  │ • Error handling│  │ • Sequence Prep │  │ • Data Retrieval│                    │
│  │ • Rate limiting │  │ • Train/Test    │  │ • Performance   │                    │
│  │                 │  │   Splitting     │  │   Tracking      │                    │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                    │
│                                                                                     │
│  Supported Instruments:                                                            │
│  • Stocks: AAPL, GOOGL, MSFT, TSLA                                                │
│  • Crypto: BTC-USD, ETH-USD                                                       │
│  • Forex: EURUSD=X (extensible architecture)                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              PERSISTENCE LAYER                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  MongoDB Database:                                                                  │
│                                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                    │
│  │ historical_     │  │   predictions   │  │model_performance│                    │
│  │    prices       │  │                 │  │                 │                    │
│  │                 │  │ • Model outputs │  │ • RMSE metrics  │                    │
│  │ • OHLCV data    │  │ • Confidence    │  │ • Training logs │                    │
│  │ • Multi-asset   │  │   intervals     │  │ • Comparison    │                    │
│  │ • Time series   │  │ • Forecast      │  │   results       │                    │
│  │ • Indexing      │  │   horizons      │  │ • Model         │                    │
│  └─────────────────┘  └─────────────────┘  │   rankings      │                    │
│                                            └─────────────────┘                    │
│  ┌─────────────────┐  ┌─────────────────┐                                         │
│  │ supported_      │  │  saved_models/  │                                         │
│  │ instruments     │  │                 │                                         │
│  │                 │  │ • .pth files    │                                         │
│  │ • Asset config  │  │ • .pkl scalers  │                                         │
│  │ • Data sources  │  │ • Model states  │                                         │
│  │ • Active flags  │  │ • Checkpoints   │                                         │
│  └─────────────────┘  └─────────────────┘                                         │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              EXTERNAL SERVICES                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  Data Sources:                                                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                    │
│  │    yfinance     │  │      ccxt       │  │    Yahoo        │                    │
│  │                 │  │                 │  │    Finance      │                    │
│  │ • Stock data    │  │ • Crypto data   │  │ • Historical    │                    │
│  │ • Real-time     │  │ • Exchange APIs │  │   prices        │                    │
│  │ • Historical    │  │ • Multiple      │  │ • Market data   │                    │
│  │ • Market data   │  │   exchanges     │  │ • Free tier     │                    │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                    │
│                                                                                     │
│  Cloud Services:                                                                   │
│  ┌─────────────────┐  ┌─────────────────┐                                         │
│  │  MongoDB Atlas  │  │  Hugging Face   │                                         │
│  │                 │  │                 │                                         │
│  │ • Cloud DB      │  │ • Model hosting │                                         │
│  │ • Automatic     │  │ • Version       │                                         │
│  │   scaling       │  │   control       │                                         │
│  │ • Backup        │  │ • Sharing       │                                         │
│  └─────────────────┘  └─────────────────┘                                         │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 🏗️ Component Architecture

### 1. **Presentation Layer**
- **Frontend Templates**: Jinja2-based HTML templates with Bootstrap 5
- **Interactive Charts**: Plotly.js for real-time candlestick charts
- **User Interface**: Responsive design with instrument/horizon selection
- **Client-side Logic**: JavaScript for dynamic updates and API calls

### 2. **Application Layer** 
- **Flask Web Framework**: Main application server
- **Route Handlers**: Web and API endpoint management
- **Request Processing**: Validation, orchestration, and response formatting
- **Session Management**: User preferences and state management

### 3. **Business Logic Layer**
- **Model Implementations**: Traditional (SMA, EMA, ARIMA, VAR) and Neural (LSTM, GRU)
- **Ensemble Methods**: 5 different combining strategies
- **Training Pipeline**: Unified model training and evaluation
- **Performance Evaluation**: Comprehensive metrics calculation

### 4. **Data Access Layer**
- **Data Collection**: Multi-source data gathering (yfinance, ccxt)
- **Data Processing**: Feature engineering and preprocessing
- **Database Operations**: MongoDB CRUD operations
- **Caching Strategy**: Intelligent data caching for performance

### 5. **Persistence Layer**
- **MongoDB Database**: Document-based storage
- **File System**: Model persistence (.pth, .pkl files)
- **Indexing Strategy**: Optimized queries for time series data
- **Backup Strategy**: Automated data protection

## 🔄 Data Flow Architecture

### 1. **Data Ingestion Flow**
```
External APIs → Data Collector → Data Processor → MongoDB → Model Training
```

### 2. **Prediction Flow**
```
User Request → Flask Route → Model Manager → Trained Models → Prediction → Database → Response
```

### 3. **Training Flow**
```
Historical Data → Feature Engineering → Model Training → Performance Evaluation → Model Storage
```

## 🛡️ Security Architecture

### Data Security
- **Input Validation**: All user inputs sanitized
- **API Rate Limiting**: Prevents abuse of external APIs
- **Error Handling**: Graceful failure without data exposure
- **Database Security**: MongoDB authentication and authorization

### Application Security
- **CSRF Protection**: Flask-WTF integration
- **SQL Injection Prevention**: Using MongoDB (NoSQL)
- **XSS Protection**: Template auto-escaping
- **Environment Variables**: Sensitive data in .env files

## 📈 Scalability Architecture

### Horizontal Scaling
- **Stateless Design**: Application can be replicated
- **Database Sharding**: MongoDB supports horizontal scaling
- **Load Balancing**: Ready for multiple Flask instances
- **Caching Strategy**: Redis integration possible

### Performance Optimization
- **Model Caching**: Pre-trained models cached in memory
- **Data Caching**: Frequently accessed data cached
- **Async Processing**: Background model training capability
- **Database Indexing**: Optimized query performance

## 🔧 Deployment Architecture

### Development Environment
- **Local Development**: SQLite fallback for offline development
- **Testing Environment**: Comprehensive test suite
- **CI/CD Ready**: GitHub Actions integration possible
- **Docker Support**: Containerization ready

### Production Environment
- **Cloud Database**: MongoDB Atlas for production
- **Model Storage**: Cloud storage for trained models
- **Monitoring**: Application performance monitoring
- **Backup Strategy**: Automated data backup and recovery

## 📊 Technology Stack Summary

| Layer | Technologies |
|-------|-------------|
| **Frontend** | HTML5, CSS3, JavaScript, Bootstrap 5, Plotly.js |
| **Backend** | Python 3.9+, Flask, Jinja2 |
| **Models** | PyTorch, scikit-learn, statsmodels |
| **Database** | MongoDB, pymongo |
| **Data** | pandas, numpy, yfinance, ccxt |
| **Testing** | unittest, pytest |
| **Deployment** | pip, requirements.txt, MongoDB Atlas |

This architecture ensures scalability, maintainability, and extensibility while meeting all assignment requirements for a production-ready financial forecasting application.
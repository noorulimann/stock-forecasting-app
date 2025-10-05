/**
 * FinanceAI Forecaster - Main JavaScript Application
 * Handles global functionality, API interactions, and UI enhancements
 */

// Global configuration
const APP_CONFIG = {
    apiBaseUrl: '/api',
    refreshInterval: 300000, // 5 minutes
    chartColors: ['#667eea', '#764ba2', '#4facfe', '#00f2fe', '#fa709a', '#fee140'],
    defaultInstrument: 'AAPL',
    dateFormat: 'YYYY-MM-DD HH:mm:ss'
};

// Global state management
const AppState = {
    currentInstrument: 'AAPL',
    activeModels: [],
    lastUpdate: null,
    connectionStatus: 'unknown'
};

// Utility functions
const Utils = {
    /**
     * Format currency values
     */
    formatCurrency: (value, currency = 'USD') => {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: currency,
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }).format(value);
    },

    /**
     * Format percentage values
     */
    formatPercentage: (value, decimals = 2) => {
        return `${value.toFixed(decimals)}%`;
    },

    /**
     * Format date/time
     */
    formatDateTime: (date) => {
        return new Date(date).toLocaleString();
    },

    /**
     * Show loading spinner
     */
    showLoading: (elementId) => {
        const element = document.getElementById(elementId);
        if (element) {
            element.style.display = 'block';
        }
    },

    /**
     * Hide loading spinner
     */
    hideLoading: (elementId) => {
        const element = document.getElementById(elementId);
        if (element) {
            element.style.display = 'none';
        }
    },

    /**
     * Show toast notification
     */
    showToast: (message, type = 'info') => {
        const toastHtml = `
            <div class="toast align-items-center text-white bg-${type} border-0" role="alert">
                <div class="d-flex">
                    <div class="toast-body">
                        ${message}
                    </div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                </div>
            </div>
        `;
        
        let toastContainer = document.getElementById('toastContainer');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.id = 'toastContainer';
            toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
            toastContainer.style.zIndex = '9999';
            document.body.appendChild(toastContainer);
        }
        
        toastContainer.insertAdjacentHTML('beforeend', toastHtml);
        const toastElement = toastContainer.lastElementChild;
        const toast = new bootstrap.Toast(toastElement);
        toast.show();
        
        // Remove toast element after hiding
        toastElement.addEventListener('hidden.bs.toast', () => {
            toastElement.remove();
        });
    },

    /**
     * Get color for model type
     */
    getModelColor: (modelName) => {
        const colorMap = {
            'SMA': '#667eea',
            'EMA': '#764ba2',
            'ARIMA': '#4facfe',
            'VAR': '#00f2fe',
            'LSTM': '#fa709a',
            'GRU': '#fee140'
        };
        return colorMap[modelName] || '#95a5a6';
    },

    /**
     * Debounce function calls
     */
    debounce: (func, wait) => {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
};

// API service
const ApiService = {
    /**
     * Generic API call wrapper
     */
    async call(endpoint, options = {}) {
        const url = `${APP_CONFIG.apiBaseUrl}${endpoint}`;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            }
        };
        
        const config = { ...defaultOptions, ...options };
        
        try {
            const response = await fetch(url, config);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            return data;
        } catch (error) {
            console.error(`API call failed for ${endpoint}:`, error);
            throw error;
        }
    },

    /**
     * Get system status
     */
    async getStatus() {
        return this.call('/status');
    },

    /**
     * Get supported instruments
     */
    async getInstruments() {
        return this.call('/instruments');
    },

    /**
     * Generate forecast
     */
    async generateForecast(data) {
        return this.call('/forecast', {
            method: 'POST',
            body: JSON.stringify(data)
        });
    },

    /**
     * Get model performance
     */
    async getPerformance() {
        return this.call('/performance');
    },

    /**
     * Train models
     */
    async trainModels(data) {
        return this.call('/train', {
            method: 'POST',
            body: JSON.stringify(data)
        });
    },

    /**
     * Get ensemble prediction
     */
    async getEnsemblePrediction(instrument, strategy) {
        return this.call(`/ensemble/predict/${instrument}`, {
            method: 'POST',
            body: JSON.stringify({ strategy })
        });
    },

    /**
     * Compare ensemble strategies
     */
    async compareEnsembleStrategies(instrument) {
        return this.call(`/ensemble/compare/${instrument}`);
    }
};

// Chart utilities
const ChartUtils = {
    /**
     * Create a basic line chart
     */
    createLineChart(elementId, data, title) {
        const traces = data.map((series, index) => ({
            x: series.x,
            y: series.y,
            type: 'scatter',
            mode: 'lines+markers',
            name: series.name,
            line: {
                color: APP_CONFIG.chartColors[index % APP_CONFIG.chartColors.length],
                width: 2
            },
            marker: {
                size: 6
            }
        }));

        const layout = {
            title: title,
            xaxis: { title: 'Time' },
            yaxis: { title: 'Price ($)' },
            showlegend: true,
            margin: { l: 50, r: 50, t: 50, b: 50 },
            plot_bgcolor: '#fafafa',
            paper_bgcolor: 'white'
        };

        const config = {
            responsive: true,
            displayModeBar: true,
            displaylogo: false
        };

        Plotly.newPlot(elementId, traces, layout, config);
    },

    /**
     * Create a bar chart
     */
    createBarChart(elementId, data, title) {
        const trace = {
            x: data.labels,
            y: data.values,
            type: 'bar',
            marker: {
                color: data.labels.map((_, index) => 
                    APP_CONFIG.chartColors[index % APP_CONFIG.chartColors.length]),
                opacity: 0.8
            }
        };

        const layout = {
            title: title,
            xaxis: { title: 'Categories' },
            yaxis: { title: 'Values' },
            margin: { l: 50, r: 50, t: 50, b: 50 }
        };

        Plotly.newPlot(elementId, [trace], layout, {responsive: true});
    },

    /**
     * Update chart data
     */
    updateChart(elementId, newData) {
        Plotly.redraw(elementId);
    }
};

// Form validation
const FormValidator = {
    /**
     * Validate forecast form
     */
    validateForecastForm(formData) {
        const errors = [];
        
        if (!formData.instrument) {
            errors.push('Please select a financial instrument');
        }
        
        if (!formData.model) {
            errors.push('Please select a forecasting model');
        }
        
        if (!formData.horizon) {
            errors.push('Please select a forecast horizon');
        }
        
        return {
            isValid: errors.length === 0,
            errors: errors
        };
    },

    /**
     * Show validation errors
     */
    showErrors(errors) {
        errors.forEach(error => {
            Utils.showToast(error, 'danger');
        });
    }
};

// Connection monitor
const ConnectionMonitor = {
    isOnline: navigator.onLine,
    
    init() {
        window.addEventListener('online', this.handleOnline.bind(this));
        window.addEventListener('offline', this.handleOffline.bind(this));
        this.checkConnection();
    },
    
    handleOnline() {
        this.isOnline = true;
        AppState.connectionStatus = 'online';
        Utils.showToast('Connection restored', 'success');
        this.updateConnectionStatus();
    },
    
    handleOffline() {
        this.isOnline = false;
        AppState.connectionStatus = 'offline';
        Utils.showToast('Connection lost', 'warning');
        this.updateConnectionStatus();
    },
    
    async checkConnection() {
        try {
            await ApiService.getStatus();
            if (!this.isOnline) {
                this.handleOnline();
            }
        } catch (error) {
            if (this.isOnline) {
                this.handleOffline();
            }
        }
    },
    
    updateConnectionStatus() {
        const statusElements = document.querySelectorAll('[data-connection-status]');
        statusElements.forEach(element => {
            element.textContent = this.isOnline ? 'Connected' : 'Disconnected';
            element.className = this.isOnline ? 
                'status-badge status-success' : 
                'status-badge status-danger';
        });
    }
};

// Auto-refresh functionality
const AutoRefresh = {
    intervals: new Map(),
    
    start(key, callback, interval = APP_CONFIG.refreshInterval) {
        this.stop(key); // Clear existing interval
        const intervalId = setInterval(callback, interval);
        this.intervals.set(key, intervalId);
    },
    
    stop(key) {
        if (this.intervals.has(key)) {
            clearInterval(this.intervals.get(key));
            this.intervals.delete(key);
        }
    },
    
    stopAll() {
        this.intervals.forEach((intervalId) => {
            clearInterval(intervalId);
        });
        this.intervals.clear();
    }
};

// Local storage management
const StorageManager = {
    set(key, value) {
        try {
            localStorage.setItem(key, JSON.stringify(value));
        } catch (error) {
            console.error('Failed to save to localStorage:', error);
        }
    },
    
    get(key, defaultValue = null) {
        try {
            const item = localStorage.getItem(key);
            return item ? JSON.parse(item) : defaultValue;
        } catch (error) {
            console.error('Failed to load from localStorage:', error);
            return defaultValue;
        }
    },
    
    remove(key) {
        try {
            localStorage.removeItem(key);
        } catch (error) {
            console.error('Failed to remove from localStorage:', error);
        }
    }
};

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 FinanceAI Forecaster initialized');
    
    // Initialize connection monitoring
    ConnectionMonitor.init();
    
    // Load saved preferences
    const savedInstrument = StorageManager.get('selectedInstrument', APP_CONFIG.defaultInstrument);
    AppState.currentInstrument = savedInstrument;
    
    // Set up global error handling
    window.addEventListener('error', function(event) {
        console.error('Global error:', event.error);
        Utils.showToast('An unexpected error occurred', 'danger');
    });
    
    // Set up unhandled promise rejection handling
    window.addEventListener('unhandledrejection', function(event) {
        console.error('Unhandled promise rejection:', event.reason);
        Utils.showToast('An unexpected error occurred', 'danger');
    });
    
    // Initialize tooltips and popovers
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Auto-save form data
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        const inputs = form.querySelectorAll('input, select, textarea');
        inputs.forEach(input => {
            input.addEventListener('change', Utils.debounce(() => {
                const formData = new FormData(form);
                const data = Object.fromEntries(formData);
                StorageManager.set(`form_${form.id}`, data);
            }, 1000));
        });
    });
});

// Clean up on page unload
window.addEventListener('beforeunload', function() {
    AutoRefresh.stopAll();
});

// Export global objects for use in other scripts
window.FinanceAI = {
    Utils,
    ApiService,
    ChartUtils,
    FormValidator,
    ConnectionMonitor,
    AutoRefresh,
    StorageManager,
    AppState,
    APP_CONFIG
};
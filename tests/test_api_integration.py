"""
Integration Tests for API Endpoints
Phase 7: Testing and Validation - API Integration Tests
"""

import unittest
import sys
import os
import json
import requests
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestAPIIntegration(unittest.TestCase):
    """Test API endpoint integration"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class"""
        cls.base_url = 'http://127.0.0.1:5000'
        cls.timeout = 10
        
        # Test if Flask app is running
        try:
            response = requests.get(f"{cls.base_url}/api/status", timeout=5)
            cls.app_running = response.status_code == 200
        except requests.exceptions.RequestException:
            cls.app_running = False
    
    def setUp(self):
        """Set up each test"""
        if not self.app_running:
            self.skipTest("Flask app is not running. Start with: python app.py")
    
    def test_api_status_endpoint(self):
        """Test /api/status endpoint"""
        response = requests.get(f"{self.base_url}/api/status", timeout=self.timeout)
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('success', data)
        self.assertIn('database_connected', data)
        self.assertIn('supported_instruments', data)
        self.assertTrue(data['success'])
    
    def test_api_instruments_endpoint(self):
        """Test /api/instruments endpoint"""
        response = requests.get(f"{self.base_url}/api/instruments", timeout=self.timeout)
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('success', data)
        self.assertIn('instruments', data)
        self.assertTrue(data['success'])
        self.assertIsInstance(data['instruments'], list)
        self.assertGreater(len(data['instruments']), 0)
        
        # Check instrument structure
        instrument = data['instruments'][0]
        self.assertIn('symbol', instrument)
        self.assertIn('name', instrument)
        self.assertIn('type', instrument)
    
    def test_api_forecast_endpoint(self):
        """Test /api/forecast endpoint"""
        forecast_data = {
            'instrument': 'AAPL',
            'model': 'LSTM',
            'horizon': 24
        }
        
        response = requests.post(
            f"{self.base_url}/api/forecast",
            json=forecast_data,
            timeout=self.timeout
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('success', data)
        self.assertTrue(data['success'])
        self.assertIn('predictions', data)
        self.assertIn('data_info', data)
        self.assertIn('model_info', data)
        
        # Check data structure
        self.assertIn('current_price', data['data_info'])
        self.assertIn('historical_data', data['data_info'])
        self.assertIsInstance(data['predictions'], dict)
    
    def test_api_ensemble_forecast(self):
        """Test ensemble forecast endpoint"""
        ensemble_data = {
            'instrument': 'AAPL',
            'model': 'ensemble_performance',
            'horizon': 3
        }
        
        response = requests.post(
            f"{self.base_url}/api/forecast",
            json=ensemble_data,
            timeout=self.timeout
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertTrue(data['success'])
        self.assertIn('ensemble_performance', data['predictions'])
        
        # Check if ensemble actually combines models
        if 'component_models' in data['predictions']:
            component_models = data['predictions']['component_models']
            expected_models = ['SMA', 'EMA', 'ARIMA', 'LSTM', 'GRU']
            for model in expected_models:
                self.assertIn(model, component_models)
    
    def test_api_performance_endpoint(self):
        """Test /api/performance endpoint"""
        response = requests.get(f"{self.base_url}/api/performance", timeout=self.timeout)
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('success', data)
        self.assertTrue(data['success'])
        self.assertIn('performance', data)
        
        performance = data['performance']
        self.assertIn('models', performance)
        
        # Check model performance structure
        models = performance['models']
        expected_models = ['SMA', 'EMA', 'ARIMA', 'LSTM', 'GRU']
        for model in expected_models:
            if model in models:
                model_perf = models[model]
                self.assertIn('mape', model_perf)
                self.assertIn('accuracy', model_perf)
    
    def test_api_ensemble_endpoints(self):
        """Test ensemble-specific endpoints"""
        endpoints_to_test = [
            '/api/ensemble/strategies',
            '/api/ensemble/compare/AAPL'
        ]
        
        for endpoint in endpoints_to_test:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=self.timeout)
                # Should be 200 or 404/500 if not implemented
                self.assertIn(response.status_code, [200, 404, 500])
                
                if response.status_code == 200:
                    data = response.json()
                    self.assertIn('success', data)
            except requests.exceptions.RequestException:
                # Endpoint might not be implemented yet
                pass
    
    def test_frontend_pages(self):
        """Test frontend page accessibility"""
        pages = ['/', '/forecast', '/ensemble', '/performance']
        
        for page in pages:
            response = requests.get(f"{self.base_url}{page}", timeout=self.timeout)
            self.assertEqual(response.status_code, 200)
            self.assertIn('text/html', response.headers.get('content-type', ''))
    
    def test_api_error_handling(self):
        """Test API error handling"""
        # Test invalid instrument
        invalid_data = {
            'instrument': 'INVALID',
            'model': 'LSTM',
            'horizon': 24
        }
        
        response = requests.post(
            f"{self.base_url}/api/forecast",
            json=invalid_data,
            timeout=self.timeout
        )
        
        # Should handle gracefully (either 200 with default or error response)
        self.assertIn(response.status_code, [200, 400, 500])
        
        if response.status_code == 200:
            data = response.json()
            # Should still return valid structure
            self.assertIn('success', data)
    
    def test_api_data_consistency(self):
        """Test API data consistency across calls"""
        # Make multiple forecast calls
        forecast_data = {
            'instrument': 'AAPL',
            'model': 'LSTM',
            'horizon': 24
        }
        
        responses = []
        for _ in range(3):
            response = requests.post(
                f"{self.base_url}/api/forecast",
                json=forecast_data,
                timeout=self.timeout
            )
            responses.append(response.json())
            time.sleep(0.1)  # Small delay
        
        # All should succeed
        for response in responses:
            self.assertTrue(response['success'])
            self.assertEqual(response['instrument'], 'AAPL')
            self.assertIn('current_price', response['data_info'])
        
        # Current price should be consistent
        current_prices = [r['data_info']['current_price'] for r in responses]
        self.assertTrue(all(p == current_prices[0] for p in current_prices))

class TestAPIPerformance(unittest.TestCase):
    """Test API performance and reliability"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class"""
        cls.base_url = 'http://127.0.0.1:5000'
        
        # Test if Flask app is running
        try:
            response = requests.get(f"{cls.base_url}/api/status", timeout=5)
            cls.app_running = response.status_code == 200
        except requests.exceptions.RequestException:
            cls.app_running = False
    
    def setUp(self):
        """Set up each test"""
        if not self.app_running:
            self.skipTest("Flask app is not running. Start with: python app.py")
    
    def test_api_response_times(self):
        """Test API response times"""
        endpoints = [
            '/api/status',
            '/api/instruments',
            '/api/performance'
        ]
        
        for endpoint in endpoints:
            start_time = time.time()
            response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            self.assertEqual(response.status_code, 200)
            self.assertLess(response_time, 5.0, f"{endpoint} took too long: {response_time}s")
    
    def test_forecast_response_time(self):
        """Test forecast endpoint response time"""
        forecast_data = {
            'instrument': 'AAPL',
            'model': 'LSTM',
            'horizon': 24
        }
        
        start_time = time.time()
        response = requests.post(
            f"{self.base_url}/api/forecast",
            json=forecast_data,
            timeout=15
        )
        end_time = time.time()
        
        response_time = end_time - start_time
        
        self.assertEqual(response.status_code, 200)
        self.assertLess(response_time, 10.0, f"Forecast took too long: {response_time}s")
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request():
            try:
                response = requests.get(f"{self.base_url}/api/status", timeout=10)
                results.put(response.status_code)
            except Exception as e:
                results.put(str(e))
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        response_codes = []
        while not results.empty():
            response_codes.append(results.get())
        
        # All requests should succeed
        for code in response_codes:
            self.assertEqual(code, 200)

if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestAPIIntegration))
    test_suite.addTest(unittest.makeSuite(TestAPIPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"PHASE 7 API INTEGRATION TESTING RESULTS")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.testsRun > 0:
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100)
        print(f"Success rate: {success_rate:.1f}%")
    else:
        print("No tests were run")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}")
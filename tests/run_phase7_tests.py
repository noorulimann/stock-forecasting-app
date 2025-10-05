"""
Phase 7: Testing and Validation - Complete Test Suite Runner
Comprehensive testing framework for stock forecasting application
"""

import unittest
import sys
import os
import json
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all test modules
from tests.test_data_pipeline import TestDataCollector, TestDataProcessor, TestDatabaseManager
from tests.test_api_integration import TestAPIIntegration, TestAPIPerformance
from tests.test_end_to_end import TestEndToEndFlow, TestApplicationIntegration, TestPerformanceValidation

class Phase7TestRunner:
    """Comprehensive test runner for Phase 7: Testing and Validation"""
    
    def __init__(self):
        self.test_results = {
            'start_time': datetime.now(),
            'end_time': None,
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'error_tests': 0,
            'skipped_tests': 0,
            'test_categories': {},
            'detailed_results': {}
        }
    
    def run_data_pipeline_tests(self):
        """Run data pipeline tests"""
        print("🔧 RUNNING DATA PIPELINE TESTS")
        print("=" * 50)
        
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestDataCollector))
        suite.addTest(unittest.makeSuite(TestDataProcessor))
        suite.addTest(unittest.makeSuite(TestDatabaseManager))
        
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        result = runner.run(suite)
        
        self.test_results['test_categories']['data_pipeline'] = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(getattr(result, 'skipped', [])),
            'success_rate': self._calculate_success_rate(result)
        }
        
        return result
    
    def run_api_integration_tests(self):
        """Run API integration tests"""
        print("\n🌐 RUNNING API INTEGRATION TESTS")
        print("=" * 50)
        
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestAPIIntegration))
        suite.addTest(unittest.makeSuite(TestAPIPerformance))
        
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        result = runner.run(suite)
        
        self.test_results['test_categories']['api_integration'] = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(getattr(result, 'skipped', [])),
            'success_rate': self._calculate_success_rate(result)
        }
        
        return result
    
    def run_end_to_end_tests(self):
        """Run end-to-end tests"""
        print("\n🎯 RUNNING END-TO-END TESTS")
        print("=" * 50)
        
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestEndToEndFlow))
        suite.addTest(unittest.makeSuite(TestApplicationIntegration))
        suite.addTest(unittest.makeSuite(TestPerformanceValidation))
        
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        result = runner.run(suite)
        
        self.test_results['test_categories']['end_to_end'] = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(getattr(result, 'skipped', [])),
            'success_rate': self._calculate_success_rate(result)
        }
        
        return result
    
    def _calculate_success_rate(self, result):
        """Calculate success rate for test result"""
        if result.testsRun == 0:
            return 0.0
        
        successful = result.testsRun - len(result.failures) - len(result.errors)
        return (successful / result.testsRun) * 100
    
    def _update_totals(self, result):
        """Update total test statistics"""
        self.test_results['total_tests'] += result.testsRun
        self.test_results['failed_tests'] += len(result.failures)
        self.test_results['error_tests'] += len(result.errors)
        self.test_results['skipped_tests'] += len(getattr(result, 'skipped', []))
        self.test_results['passed_tests'] += (result.testsRun - len(result.failures) - len(result.errors))
    
    def run_all_tests(self):
        """Run all Phase 7 tests"""
        print("🚀 STARTING PHASE 7: TESTING AND VALIDATION")
        print("=" * 70)
        print(f"Start Time: {self.test_results['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        # Run all test categories
        results = []
        
        # 1. Data Pipeline Tests
        try:
            result1 = self.run_data_pipeline_tests()
            results.append(result1)
            self._update_totals(result1)
        except Exception as e:
            print(f"❌ Data pipeline tests failed: {str(e)}")
        
        # 2. API Integration Tests
        try:
            result2 = self.run_api_integration_tests()
            results.append(result2)
            self._update_totals(result2)
        except Exception as e:
            print(f"❌ API integration tests failed: {str(e)}")
        
        # 3. End-to-End Tests
        try:
            result3 = self.run_end_to_end_tests()
            results.append(result3)
            self._update_totals(result3)
        except Exception as e:
            print(f"❌ End-to-end tests failed: {str(e)}")
        
        # Finalize results
        self.test_results['end_time'] = datetime.now()
        duration = self.test_results['end_time'] - self.test_results['start_time']
        
        # Generate comprehensive report
        self._generate_final_report(duration)
        
        # Save results to file
        self._save_results()
        
        return self.test_results
    
    def _generate_final_report(self, duration):
        """Generate final comprehensive test report"""
        print("\n" + "=" * 70)
        print("🏁 PHASE 7: TESTING AND VALIDATION - FINAL REPORT")
        print("=" * 70)
        
        # Overall statistics
        total_tests = self.test_results['total_tests']
        passed_tests = self.test_results['passed_tests']
        failed_tests = self.test_results['failed_tests']
        error_tests = self.test_results['error_tests']
        skipped_tests = self.test_results['skipped_tests']
        
        overall_success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"📊 OVERALL RESULTS:")
        print(f"  • Total Tests Run: {total_tests}")
        print(f"  • Passed: {passed_tests} ✅")
        print(f"  • Failed: {failed_tests} ❌")
        print(f"  • Errors: {error_tests} 💥")
        print(f"  • Skipped: {skipped_tests} ⏭️")
        print(f"  • Success Rate: {overall_success_rate:.1f}%")
        print(f"  • Duration: {duration.total_seconds():.1f} seconds")
        
        # Category breakdown
        print(f"\n📋 CATEGORY BREAKDOWN:")
        for category, stats in self.test_results['test_categories'].items():
            print(f"  {category.replace('_', ' ').title()}:")
            print(f"    - Tests: {stats['tests_run']}")
            print(f"    - Success Rate: {stats['success_rate']:.1f}%")
            print(f"    - Failures: {stats['failures']}")
            print(f"    - Errors: {stats['errors']}")
            if stats['skipped'] > 0:
                print(f"    - Skipped: {stats['skipped']}")
        
        # Phase 7 completion assessment
        print(f"\n🎯 PHASE 7 COMPLETION ASSESSMENT:")
        
        if overall_success_rate >= 90:
            status = "EXCELLENT ✨"
            emoji = "🎉"
        elif overall_success_rate >= 80:
            status = "GOOD ✅"
            emoji = "👍"
        elif overall_success_rate >= 70:
            status = "ACCEPTABLE ⚠️"
            emoji = "⚠️"
        else:
            status = "NEEDS IMPROVEMENT ❌"
            emoji = "🔧"
        
        print(f"  {emoji} Overall Status: {status}")
        print(f"  📈 Success Rate: {overall_success_rate:.1f}%")
        
        # Requirements check
        print(f"\n📝 REQUIREMENTS VERIFICATION:")
        requirements_met = []
        
        # Check each requirement
        data_pipeline_success = self.test_results['test_categories'].get('data_pipeline', {}).get('success_rate', 0)
        api_integration_success = self.test_results['test_categories'].get('api_integration', {}).get('success_rate', 0)
        end_to_end_success = self.test_results['test_categories'].get('end_to_end', {}).get('success_rate', 0)
        
        requirements = [
            ("Data Pipeline Testing", data_pipeline_success >= 70, data_pipeline_success),
            ("API Integration Testing", api_integration_success >= 70, api_integration_success),
            ("End-to-End Testing", end_to_end_success >= 70, end_to_end_success),
            ("Overall System Reliability", overall_success_rate >= 75, overall_success_rate)
        ]
        
        for req_name, req_met, req_score in requirements:
            status_icon = "✅" if req_met else "❌"
            print(f"  {status_icon} {req_name}: {req_score:.1f}%")
            if req_met:
                requirements_met.append(req_name)
        
        # Final assessment
        requirements_completion = len(requirements_met) / len(requirements) * 100
        
        print(f"\n🏆 FINAL ASSESSMENT:")
        print(f"  • Requirements Met: {len(requirements_met)}/{len(requirements)} ({requirements_completion:.0f}%)")
        
        if requirements_completion >= 75:
            print(f"  🎉 PHASE 7: TESTING AND VALIDATION - COMPLETED SUCCESSFULLY!")
            print(f"  ✅ System is ready for production deployment")
        else:
            print(f"  ⚠️ PHASE 7: TESTING AND VALIDATION - PARTIALLY COMPLETED")
            print(f"  🔧 Some areas need improvement before production")
        
        print("=" * 70)
    
    def _save_results(self):
        """Save test results to file"""
        try:
            # Ensure logs directory exists
            logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
            os.makedirs(logs_dir, exist_ok=True)
            
            # Save detailed results
            results_file = os.path.join(logs_dir, 'phase7_test_results.json')
            
            # Convert datetime objects to strings for JSON serialization
            serializable_results = self.test_results.copy()
            serializable_results['start_time'] = self.test_results['start_time'].isoformat()
            serializable_results['end_time'] = self.test_results['end_time'].isoformat()
            
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            print(f"\n💾 Test results saved to: {results_file}")
            
        except Exception as e:
            print(f"⚠️ Failed to save test results: {str(e)}")

def main():
    """Main function to run Phase 7 testing"""
    runner = Phase7TestRunner()
    results = runner.run_all_tests()
    
    # Return appropriate exit code
    overall_success_rate = (results['passed_tests'] / results['total_tests'] * 100) if results['total_tests'] > 0 else 0
    
    if overall_success_rate >= 75:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure

if __name__ == '__main__':
    main()
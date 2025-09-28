#!/usr/bin/env python3
"""
Master Test Runner - REAL DATA ONLY
Runs all comprehensive tests to verify the treasury risk dashboard works with real data.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MasterTestRunner:
    """Comprehensive test runner for all system components."""

    def __init__(self):
        self.test_results = {}
        self.start_time = datetime.now()

    def run_test_file(self, test_file, description):
        """Run a single test file and capture results."""
        logger.info(f"\n{'='*80}")
        logger.info(f"RUNNING: {description}")
        logger.info(f"FILE: {test_file}")
        logger.info(f"{'='*80}")

        try:
            # Run the test file
            result = subprocess.run(
                [sys.executable, test_file],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            # Log output
            if result.stdout:
                logger.info("STDOUT:")
                for line in result.stdout.strip().split('\n'):
                    logger.info(f"  {line}")

            if result.stderr and result.returncode != 0:
                logger.error("STDERR:")
                for line in result.stderr.strip().split('\n'):
                    logger.error(f"  {line}")

            # Check result
            success = result.returncode == 0
            self.test_results[description] = success

            if success:
                logger.info(f"✅ {description}: PASSED")
            else:
                logger.error(f"❌ {description}: FAILED (exit code: {result.returncode})")

            return success

        except subprocess.TimeoutExpired:
            logger.error(f"❌ {description}: TIMEOUT (>5 minutes)")
            self.test_results[description] = False
            return False

        except Exception as e:
            logger.error(f"❌ {description}: ERROR - {e}")
            self.test_results[description] = False
            return False

    async def verify_no_mock_data(self):
        """Verify no mock data exists in the codebase."""
        logger.info(f"\n{'='*80}")
        logger.info("VERIFYING NO MOCK DATA IN CODEBASE")
        logger.info(f"{'='*80}")

        mock_indicators = [
            "mock", "fake", "synthetic", "simulated", "generated",
            "random.uniform", "random.randint", "dummy", "placeholder",
            "lorem ipsum", "test data", "sample data"
        ]

        backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
        suspicious_files = []

        if os.path.exists(backend_dir):
            for root, dirs, files in os.walk(backend_dir):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read().lower()
                                for indicator in mock_indicators:
                                    if indicator in content and 'test' not in file.lower():
                                        suspicious_files.append((file_path, indicator))
                                        break
                        except Exception as e:
                            logger.warning(f"Could not read {file_path}: {e}")

        if suspicious_files:
            logger.warning("Potential mock data indicators found:")
            for file_path, indicator in suspicious_files:
                logger.warning(f"  {file_path}: '{indicator}'")
            self.test_results["No Mock Data"] = False
        else:
            logger.info("✅ No mock data indicators found in codebase")
            self.test_results["No Mock Data"] = True

        return len(suspicious_files) == 0

    def verify_real_api_endpoints(self):
        """Verify all API endpoints are real."""
        logger.info(f"\n{'='*80}")
        logger.info("VERIFYING REAL API ENDPOINTS")
        logger.info(f"{'='*80}")

        real_endpoints = [
            "https://api.dexscreener.com",
            "https://api.binance.com",
            "https://api.exchange.coinbase.com"
        ]

        config_files = [
            os.path.join('backend', 'app', 'utils', 'config.py'),
            os.path.join('backend', 'app', 'services', 'dexscreener_client.py'),
            os.path.join('backend', 'app', 'services', 'ingestion.py')
        ]

        verified_endpoints = []

        for config_file in config_files:
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        for endpoint in real_endpoints:
                            if endpoint in content:
                                verified_endpoints.append(endpoint)
                                logger.info(f"  ✅ Found {endpoint} in {config_file}")
                except Exception as e:
                    logger.warning(f"Could not read {config_file}: {e}")

        unique_endpoints = list(set(verified_endpoints))
        all_verified = len(unique_endpoints) >= len(real_endpoints)

        if all_verified:
            logger.info("✅ All real API endpoints verified in configuration")
            self.test_results["Real API Endpoints"] = True
        else:
            logger.warning(f"⚠️ Only {len(unique_endpoints)}/{len(real_endpoints)} endpoints verified")
            self.test_results["Real API Endpoints"] = False

        return all_verified

    def check_dependencies(self):
        """Check if all required dependencies are available."""
        logger.info(f"\n{'='*80}")
        logger.info("CHECKING DEPENDENCIES")
        logger.info(f"{'='*80}")

        required_packages = ['aiohttp', 'pandas', 'yfinance', 'pydantic']
        available_packages = []

        for package in required_packages:
            try:
                __import__(package)
                available_packages.append(package)
                logger.info(f"  ✅ {package}: Available")
            except ImportError:
                logger.warning(f"  ⚠️ {package}: Not available")

        all_available = len(available_packages) == len(required_packages)
        self.test_results["Dependencies"] = all_available

        return all_available

    def run_all_tests(self):
        """Run all comprehensive tests."""
        logger.info("🚀 STARTING COMPREHENSIVE TREASURY DASHBOARD TESTS")
        logger.info(f"Start time: {self.start_time}")
        logger.info("=" * 100)

        # Check dependencies first
        if not self.check_dependencies():
            logger.error("❌ Missing dependencies - tests may fail")

        # Verify real API endpoints
        self.verify_real_api_endpoints()

        # Run comprehensive test files
        test_files = [
            ("tests/test_dexscreener_real_data.py", "DexScreener API Integration Tests"),
            ("tests/test_data_ingestion_real.py", "Data Ingestion Pipeline Tests"),
            ("tests/test_end_to_end_real.py", "End-to-End System Tests"),
        ]

        for test_file, description in test_files:
            if os.path.exists(test_file):
                self.run_test_file(test_file, description)
            else:
                logger.warning(f"⚠️ Test file not found: {test_file}")
                self.test_results[description] = False

        # Verify no mock data
        asyncio.run(self.verify_no_mock_data())

        # Print final summary
        self.print_final_summary()

        return all(self.test_results.values())

    def print_final_summary(self):
        """Print comprehensive final summary."""
        end_time = datetime.now()
        duration = end_time - self.start_time

        logger.info("\n" + "=" * 100)
        logger.info("COMPREHENSIVE TEST SUITE SUMMARY")
        logger.info("=" * 100)

        passed = sum(1 for result in self.test_results.values() if result)
        total = len(self.test_results)

        logger.info(f"Execution time: {duration}")
        logger.info(f"Tests completed: {total}")
        logger.info(f"Tests passed: {passed}")
        logger.info(f"Tests failed: {total - passed}")
        logger.info(f"Success rate: {passed/total*100:.1f}%")

        logger.info("\nDETAILED RESULTS:")
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"  {test_name}: {status}")

        if passed == total:
            logger.info("\n🎉 ALL TESTS PASSED - SYSTEM READY FOR PRODUCTION!")
            logger.info("✅ DexScreener integration working with real data")
            logger.info("✅ Data ingestion pipeline operational")
            logger.info("✅ End-to-end system functioning")
            logger.info("✅ No mock data detected")
            logger.info("✅ Real API endpoints verified")
            logger.info("✅ Treasury risk dashboard fully validated")

            logger.info("\nSYSTEM CAPABILITIES VERIFIED:")
            logger.info("  • Real-time price data from DexScreener")
            logger.info("  • Historical data from Binance/Coinbase")
            logger.info("  • Risk calculations with real market data")
            logger.info("  • Forecasting capabilities")
            logger.info("  • Portfolio analytics")
            logger.info("  • Complete data pipeline integration")

        else:
            logger.warning(f"\n⚠️ {total - passed} TESTS FAILED")
            logger.warning("System requires attention before production deployment")

            failed_tests = [name for name, result in self.test_results.items() if not result]
            logger.warning("Failed tests:")
            for test in failed_tests:
                logger.warning(f"  • {test}")

        logger.info("\n" + "=" * 100)
        logger.info("TEST SUITE COMPLETED")
        logger.info("=" * 100)


def main():
    """Main test runner entry point."""
    runner = MasterTestRunner()
    success = runner.run_all_tests()
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
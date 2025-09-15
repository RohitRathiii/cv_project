"""
Test runner for all unit tests

This script runs all unit tests and provides a comprehensive test report.
"""

import unittest
import sys
import os
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def discover_and_run_tests():
    """Discover and run all unit tests"""
    # Get the test directory
    test_dir = Path(__file__).parent
    
    # Discover all test modules
    loader = unittest.TestLoader()
    suite = loader.discover(str(test_dir), pattern='test_*.py')
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        buffer=True,
        failfast=False
    )
    
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("UNIT TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors))/result.testsRun*100:.1f}%")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"- {test}")
    
    if result.skipped:
        print(f"\nSKIPPED ({len(result.skipped)}):")
        for test, reason in result.skipped:
            print(f"- {test}: {reason}")
    
    print("="*60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = discover_and_run_tests()
    sys.exit(0 if success else 1)
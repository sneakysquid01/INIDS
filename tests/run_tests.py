"""
Test Runner for INIDS Week 1-2 Test Suite
Provides easy commands to run different test configurations
"""

import subprocess
import sys
from pathlib import Path


class TestRunner:
    """Test execution utility."""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.repo_root = self.test_dir.parent
    
    def run_all_tests(self, verbose=True, coverage=False):
        """Run all tests."""
        cmd = ["pytest", str(self.test_dir)]
        if verbose:
            cmd.append("-v")
        if coverage:
            cmd.extend(["--cov=src", "--cov-report=html"])
        return subprocess.run(cmd, cwd=self.repo_root)
    
    def run_unit_tests(self, verbose=True):
        """Run only unit tests."""
        cmd = ["pytest", "-m", "unit", str(self.test_dir)]
        if verbose:
            cmd.append("-v")
        return subprocess.run(cmd, cwd=self.repo_root)
    
    def run_integration_tests(self, verbose=True):
        """Run only integration tests."""
        cmd = ["pytest", "-m", "integration", str(self.test_dir)]
        if verbose:
            cmd.append("-v")
        return subprocess.run(cmd, cwd=self.repo_root)
    
    def run_api_tests(self, verbose=True):
        """Run only API endpoint tests."""
        cmd = ["pytest", "-m", "api", str(self.test_dir)]
        if verbose:
            cmd.append("-v")
        return subprocess.run(cmd, cwd=self.repo_root)
    
    def run_week1_tests(self, verbose=True):
        """Run Week 1 feature tests."""
        cmd = ["pytest", "test_week1_features.py", "-v" if verbose else ""]
        cmd = [c for c in cmd if c]
        return subprocess.run(cmd, cwd=self.test_dir)
    
    def run_with_coverage(self, verbose=True):
        """Run tests with coverage report."""
        cmd = ["pytest", str(self.test_dir)]
        if verbose:
            cmd.append("-v")
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
        return subprocess.run(cmd, cwd=self.repo_root)
    
    def run_performance_tests(self, verbose=True):
        """Run performance tests."""
        cmd = ["pytest", "-m", "slow", str(self.test_dir)]
        if verbose:
            cmd.append("-v")
        return subprocess.run(cmd, cwd=self.repo_root)
    
    def run_specific_test(self, test_name, verbose=True):
        """Run a specific test."""
        cmd = ["pytest", "-k", test_name, str(self.test_dir)]
        if verbose:
            cmd.append("-v")
        return subprocess.run(cmd, cwd=self.repo_root)
    
    def run_all_with_markers(self, verbose=True):
        """Run all tests with detailed markers output."""
        cmd = ["pytest", str(self.test_dir), "-v", "-m", "unit or integration or api"]
        return subprocess.run(cmd, cwd=self.repo_root)


if __name__ == "__main__":
    runner = TestRunner()
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == "all":
            runner.run_all_tests(coverage=True)
        elif cmd == "unit":
            runner.run_unit_tests()
        elif cmd == "integration":
            runner.run_integration_tests()
        elif cmd == "api":
            runner.run_api_tests()
        elif cmd == "week1":
            runner.run_week1_tests()
        elif cmd == "coverage":
            runner.run_with_coverage()
        elif cmd == "perf":
            runner.run_performance_tests()
        elif cmd == "quick":
            # Quick tests without slow/performance tests
            runner.run_all_tests(coverage=False)
        else:
            print(f"Unknown command: {cmd}")
            print("\nAvailable commands:")
            print("  all          - Run all tests with coverage")
            print("  unit         - Run unit tests only")
            print("  integration  - Run integration tests only")
            print("  api          - Run API endpoint tests only")
            print("  week1        - Run Week 1 feature tests")
            print("  coverage     - Run tests with HTML coverage report")
            print("  perf         - Run performance tests")
            print("  quick        - Run quick tests (no slow tests)")
            sys.exit(1)
    else:
        # Default: run all tests
        runner.run_all_tests(coverage=True)

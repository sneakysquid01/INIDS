# INIDS Test Suite Documentation

## Overview

Complete test suite for INIDS Week 1-2 features covering:
- **Week 1 Features** (5 components): Unit tests for individual components
- **Integration Tests**: End-to-end alert flow, database operations, pipeline integration
- **API Endpoint Tests**: REST endpoint validation and response structures

---

## Test Structure

### Files

```
tests/
├── conftest.py                  # Pytest configuration and shared fixtures
├── test_week1_features.py       # Unit tests for Week 1 components
├── test_integration.py          # Integration tests for complete pipelines
├── test_api_endpoints.py        # API endpoint structure and validation tests
├── requirements-test.txt        # Test dependencies
├── run_tests.py                # Test runner script
└── README.md                    # This file
```

### Test Organization

Tests are organized by markers:
- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests (slower, multi-component)
- `@pytest.mark.api` - API endpoint tests
- `@pytest.mark.slow` - Performance/slow tests

---

## Installation

### 1. Install Test Dependencies

```bash
pip install -r tests/requirements-test.txt
```

### 2. Verify Installation

```bash
pytest --version
```

---

## Running Tests

### Quick Start

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run only unit tests (fast)
pytest -m unit tests/

# Run only integration tests
pytest -m integration tests/

# Run only API tests
pytest -m api tests/
```

### Using Test Runner Script

```bash
cd tests

# All tests with coverage
python run_tests.py all

# Unit tests only
python run_tests.py unit

# Integration tests
python run_tests.py integration

# API tests
python run_tests.py api

# Week 1 features
python run_tests.py week1

# Performance tests
python run_tests.py perf

# Quick tests (no slow tests)
python run_tests.py quick
```

---

## Test Suite Details

### 1. Week 1 Feature Tests (`test_week1_features.py`)

#### Honeypot Detection Engine
- [x] Import test
- [x] Initialization test
- [x] Canary IP detection
- [x] Normal traffic exclusion
- [x] Dynamic IP updates

**Run:**
```bash
pytest tests/test_week1_features.py::TestHoneypotDetectionEngine -v
```

#### Enhanced Rule Compiler
- [x] Import and basic compilation
- [x] Regex operator support
- [x] Range operator support
- [x] AND/OR logic operators
- [x] Performance with 100+ rules

**Run:**
```bash
pytest tests/test_week1_features.py::TestEnhancedRuleCompiler -v
```

#### Hierarchical Incident Aggregation
- [x] Import and initialization
- [x] Single alert aggregation
- [x] Multiple alerts from same IP
- [x] Incident retrieval
- [x] Activity grouping

**Run:**
```bash
pytest tests/test_week1_features.py::TestHierarchicalIncidentAggregation -v
```

#### Temporal Correlation Engine
- [x] Import and initialization
- [x] Pattern registration
- [x] No match scenario
- [x] Multi-stage attack detection
- [x] Time window validation

**Run:**
```bash
pytest tests/test_week1_features.py::TestTemporalCorrelationEngine -v
```

#### Config Manager
- [x] Import test
- [x] Instance creation

**Run:**
```bash
pytest tests/test_week1_features.py::TestConfigManager -v
```

---

### 2. Integration Tests (`test_integration.py`)

#### Alert Filtering Pipeline
- [x] EXCLUDE layer (blocks completely)
- [x] IGNORE layer (deprioritizes)
- [x] MERGE layer (combines similar)

**Run:**
```bash
pytest tests/test_integration.py::TestAlertFilteringPipeline -v
```

#### Entity Enrichment Pipeline
- [x] Basic enrichment
- [x] Threat level assessment
- [x] Internal IP detection

**Run:**
```bash
pytest tests/test_integration.py::TestEntityEnrichmentPipeline -v
```

#### Incident Aggregation Pipeline
- [x] Single source IP aggregation
- [x] Multiple IPs create separate incidents

**Run:**
```bash
pytest tests/test_integration.py::TestIncidentAggregationPipeline -v
```

#### Temporal Correlation Pipeline
- [x] Sequential attack pattern detection
- [x] Time window violation handling

**Run:**
```bash
pytest tests/test_integration.py::TestTemporalCorrelationPipeline -v
```

#### End-to-End Alert Flow
- [x] Complete lifecycle: filter → enrich → aggregate → correlate

**Run:**
```bash
pytest tests/test_integration.py::TestEndToEndAlertFlow -v
```

#### Database Operations
- [x] Save and retrieve alerts
- [x] Schema validation

**Run:**
```bash
pytest tests/test_integration.py::TestDatabaseOperations -v
```

---

### 3. API Endpoint Tests (`test_api_endpoints.py`)

#### Honeypot API
- [x] Endpoint existence
- [x] IP format validation
- [x] Port format validation

#### Incident API
- [x] Endpoint existence
- [x] Response structures

#### Temporal API
- [x] Patterns endpoint
- [x] State endpoint
- [x] Pattern structure validation

#### Entity Enrichment API
- [x] Enrichment endpoint
- [x] Threat level endpoint
- [x] Response structures
- [x] Confidence scoring

#### Alert Filtering API
- [x] All filter endpoints
- [x] Rule request structures
- [x] Statistics response format

#### HTTP Status Codes
- [x] GET returns 200
- [x] POST returns 201
- [x] Deletion returns 200
- [x] Not found returns 404
- [x] Bad request returns 400

#### Authorization
- [x] Analyst role endpoints
- [x] Admin role endpoints

#### Error Handling
- [x] Error response structure
- [x] Success response structure

---

## Test Fixtures

### Available Fixtures (in `conftest.py`)

```python
@pytest.fixture
def temp_db()                    # Temporary SQLite database
    → yields db_path

@pytest.fixture
def mock_ops_store(temp_db)     # Mock OpsStore instance
    → mock database operations

@pytest.fixture
def sample_alert()              # Sample alert data
    → {id, timestamp, severity, ...}

@pytest.fixture
def sample_rules()              # Sample filter rules
    → {exclude, ignore, merge}

@pytest.fixture
def honeypot_config()           # Honeypot configuration
    → {honeypot_ips, honeypot_ports, enabled}
```

---

## Expected Test Results

### Summary
- **Total Tests**: 70+
- **Unit Tests**: 35+ (fast, < 1 second)
- **Integration Tests**: 25+ (medium, 1-5 seconds)
- **API Tests**: 40+ (validation, < 1 second)
- **Performance Tests**: 3 (slow, 5-30 seconds)

### Coverage Targets
- **Overall**: 80%+
- **Core Detection**: 85%+
- **Filtering Engine**: 90%+
- **Enrichment Engine**: 85%+

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, "3.10", "3.11"]
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -q -r requirements.txt
        pip install -q -r tests/requirements-test.txt
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=src --cov-report=term-missing
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

---

## Troubleshooting

### Issue: Tests fail with "ModuleNotFoundError"

**Solution**: Ensure you're in the repo root directory and install dependencies:
```bash
pip install -r requirements.txt
pip install -r tests/requirements-test.txt
```

### Issue: Database tests fail with permission error

**Solution**: Ensure temp directory is writable:
```bash
# On Unix
chmod 777 /tmp

# On Windows, use system temp
# Tests use tempfile module automatically
```

### Issue: Redis config tests fail

**Solution**: This is expected if Redis is not running. The tests handle this gracefully.

### Issue: Performance tests timeout

**Solution**: Use `pytest -m "not slow"` to skip slow tests during development.

---

## Development Guidelines

### Adding New Tests

1. Choose appropriate test file:
   - Unit tests → `test_week1_features.py`
   - Integration tests → `test_integration.py`
   - API tests → `test_api_endpoints.py`

2. Add test function with marker:
```python
@pytest.mark.unit
def test_my_feature():
    """Test description."""
    # Arrange
    obj = MyComponent()
    
    # Act
    result = obj.do_something()
    
    # Assert
    assert result == expected
```

3. Run new test:
```bash
pytest tests/test_xxxx.py::test_my_feature -v
```

### Test Naming Convention

- `test_<component>_<scenario>` - Descriptive names
- Use underscores, not camelCase
- Include the aspect being tested

**Good:** `test_filter_alert_exclude_layer`
**Bad:** `testFilterExclude`

---

## Performance Benchmarks

Expected performance (single-threaded):

```
Honeypot Detection:     ~100 checks/second
Rule Compilation:       ~50 rules/second
Alert Filtering:        ~1000 alerts/second
Enrichment:             ~100 IPs/second
Aggregation:            ~500 alerts/second
Correlation:            ~200 patterns/second
```

---

## Continuous Integration

### Pre-commit Checks

Run before committing:
```bash
pytest tests/ -m "not slow" -q
```

### Pre-push Checks

Run before pushing to main:
```bash
pytest tests/ --cov=src -q
coverage report --fail-under=80
```

---

## Support and Issues

### Getting Help

1. Check test output: `pytest -v` shows detailed failure info
2. Use `-s` flag to see print statements: `pytest -s`
3. Run single test in isolation: `pytest tests/test_xxxx.py::test_name -v`

### Reporting Issues

When reporting test failures:
1. Provide test name and command used
2. Include full error output
3. Note Python version, OS, environment
4. Specify if it's consistent or intermittent

---

## Next Steps

After test suite passes:

1. **Code Coverage Review**: `htmlcov/index.html`
2. **Performance Profiling**: Add benchmarks for production
3. **Load Testing**: Test with production-like traffic volumes
4. **Integration Testing**: Test with real detection engines
5. **Deployment Validation**: Run acceptance tests on staging

---

*Generated: 2026-04-15 | INIDS Test Suite v1.0*

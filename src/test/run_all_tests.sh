#!/bin/bash
# Run all GSwarm tests

set -e

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test results
PASSED=0
FAILED=0
SKIPPED=0

# Function to print colored output
print_header() {
    echo -e "\n${BLUE}===================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Function to run a test category
run_test_category() {
    local category=$1
    local test_cmd=$2
    local skip_var=$3
    
    print_header "Running $category Tests"
    
    if [ ! -z "$skip_var" ] && [ "${!skip_var}" == "true" ]; then
        print_warning "$category tests skipped (set $skip_var=false to run)"
        SKIPPED=$((SKIPPED + 1))
        return
    fi
    
    cd "$category"
    if eval "$test_cmd"; then
        print_success "$category tests passed"
        PASSED=$((PASSED + 1))
    else
        print_error "$category tests failed"
        FAILED=$((FAILED + 1))
    fi
    cd ..
}

# Main test execution
print_header "GSwarm Test Suite"
echo "Starting comprehensive test run..."
echo "Base directory: $(pwd)"

# Check Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not found"
    exit 1
fi

# Unit Tests
run_test_category "unit" "python3 -m unittest discover -s . -p 'test_*.py'" ""

# Integration Tests
run_test_category "integration" "python3 test_gswarm.py" ""

# API Tests (optional)
run_test_category "api" "python3 test_rest_api.py" "SKIP_API_TESTS"

# Performance Tests (optional)
if [ "$RUN_PERFORMANCE_TESTS" == "true" ]; then
    run_test_category "performance" "python3 test_stress.py" ""
else
    print_warning "Performance tests skipped (set RUN_PERFORMANCE_TESTS=true to run)"
    SKIPPED=$((SKIPPED + 1))
fi

# Summary
print_header "Test Summary"
echo -e "Total test categories: $((PASSED + FAILED + SKIPPED))"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo -e "${YELLOW}Skipped: $SKIPPED${NC}"

if [ $FAILED -eq 0 ]; then
    print_success "All tests passed!"
    exit 0
else
    print_error "Some tests failed!"
    exit 1
fi
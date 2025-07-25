#!/bin/bash
# Run all new GSwarm tests

echo "========================================="
echo "Running GSwarm Test Suite"
echo "========================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Check gswarm is available
if ! command -v gswarm &> /dev/null; then
    echo -e "${RED}Error: gswarm command not found. Please ensure GSwarm is installed.${NC}"
    echo "You may need to activate your virtual environment or install gswarm."
    exit 1
fi

echo "Using Python: $(which python)"
echo "Using GSwarm: $(which gswarm)"
echo "Current PATH: $PATH"

# Function to run a test
run_test() {
    local test_name=$1
    local test_path=$2
    
    echo -e "\n${YELLOW}Running $test_name...${NC}"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    echo `which python`
    # Run python with current environment PATH
    if env PATH="$PATH" python "$test_path"; then
        echo -e "${GREEN}✓ $test_name PASSED${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}✗ $test_name FAILED${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
}

# Check if gswarm is available
if ! command -v gswarm &> /dev/null; then
    echo -e "${RED}Error: gswarm command not found. Please ensure GSwarm is installed.${NC}"
    echo "You may need to activate your virtual environment or install gswarm."
    exit 1
fi

# Change to test directory
cd "$(dirname "$0")" || exit 1

# Run tests in order
echo "Starting test execution..."

# 1. Basic functionality test
run_test "Basic Functionality Test" "basic/test_basic_functionality.py"

# 2. Profiler test
run_test "Profiler Test" "profiler/test_profiler.py"

# 3. Model serve test (may take longer)
run_test "Model Serve Test" "model/test_model_serve.py"

# 4. Prediction test
run_test "Prediction Test" "prediction/test_prediction.py"

# 5. Data handling test
run_test "Data Handling Test" "data/test_data_handling.py"

# Summary
echo -e "\n========================================="
echo "Test Summary"
echo "========================================="
echo -e "Total Tests: $TOTAL_TESTS"
echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
echo -e "${RED}Failed: $FAILED_TESTS${NC}"

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "\n${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "\n${RED}Some tests failed. Please check the output above.${NC}"
    exit 1
fi
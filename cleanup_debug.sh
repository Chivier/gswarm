#!/bin/bash

# Script to clean up debug files from the GSwarm profiler project

set -e

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Function to print colored output
print_header() {
    echo -e "\n${BLUE}===================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Dry run mode
DRY_RUN=false
if [ "$1" == "--dry-run" ] || [ "$1" == "-n" ]; then
    DRY_RUN=true
    print_warning "DRY RUN MODE - No files will be deleted"
fi

# Function to remove files
remove_files() {
    local pattern=$1
    local description=$2
    
    echo -e "\n${YELLOW}Cleaning $description...${NC}"
    
    # Find files matching pattern
    files=$(find . -name "$pattern" -type f 2>/dev/null | grep -v "/.git/" || true)
    
    if [ -z "$files" ]; then
        print_success "No $description found"
        return
    fi
    
    # Count files
    count=$(echo "$files" | wc -l)
    echo "Found $count files:"
    
    # Show files (limit to first 10 in display)
    if [ $count -le 10 ]; then
        echo "$files" | sed 's/^/  /'
    else
        echo "$files" | head -10 | sed 's/^/  /'
        echo "  ... and $((count - 10)) more"
    fi
    
    # Remove files
    if [ "$DRY_RUN" == "false" ]; then
        echo "$files" | xargs -r rm -f
        print_success "Removed $count $description"
    else
        print_warning "Would remove $count $description"
    fi
}

# Function to remove directories
remove_directories() {
    local pattern=$1
    local description=$2
    
    echo -e "\n${YELLOW}Cleaning $description...${NC}"
    
    # Find directories matching pattern
    dirs=$(find . -name "$pattern" -type d 2>/dev/null | grep -v "/.git/" || true)
    
    if [ -z "$dirs" ]; then
        print_success "No $description found"
        return
    fi
    
    # Count directories
    count=$(echo "$dirs" | wc -l)
    echo "Found $count directories:"
    
    # Show directories (limit to first 10 in display)
    if [ $count -le 10 ]; then
        echo "$dirs" | sed 's/^/  /'
    else
        echo "$dirs" | head -10 | sed 's/^/  /'
        echo "  ... and $((count - 10)) more"
    fi
    
    # Remove directories
    if [ "$DRY_RUN" == "false" ]; then
        echo "$dirs" | xargs -r rm -rf
        print_success "Removed $count $description"
    else
        print_warning "Would remove $count $description"
    fi
}

# Main cleanup
print_header "GSwarm Debug Files Cleanup"

# Change to project root
cd "$(dirname "$0")"
echo "Working directory: $(pwd)"

# Clean up log files
remove_files "*.log" "log files"

# Clean up Python cache
remove_files "*.pyc" "Python bytecode files"
remove_files "*.pyo" "Python optimized bytecode files"
remove_directories "__pycache__" "Python cache directories"
remove_directories ".pytest_cache" "pytest cache directories"

# Clean up coverage reports
remove_files ".coverage" "coverage data files"
remove_files "*.cover" "coverage report files"
remove_directories "htmlcov" "HTML coverage report directories"

# Clean up temporary files
remove_files "*.tmp" "temporary files"
remove_files "*.temp" "temporary files"
remove_files "*.swp" "vim swap files"
remove_files "*.swo" "vim swap files"
remove_files "*~" "backup files"

# Clean up debug output files
remove_files "*.debug" "debug output files"
remove_files "*.out" "output files"

# Clean up profile data
remove_files "*.prof" "profiling data files"
remove_files "*.profile" "profiling data files"

# Clean up core dumps
remove_files "core.*" "core dump files"

# Clean up specific test output files
echo -e "\n${YELLOW}Cleaning test output files...${NC}"
if [ "$DRY_RUN" == "false" ]; then
    # Remove JSON files from specific test directories
    find ./src/gswarm-standalone-scheduler/scheduler_test_v1 -name "*.json" -type f -delete 2>/dev/null || true
    find ./src/gswarm-standalone-scheduler/scheduler_test_v2 -name "*.json" -type f -delete 2>/dev/null || true
    print_success "Cleaned test output JSON files"
else
    json_count=$(find ./src/gswarm-standalone-scheduler/scheduler_test_v1 -name "*.json" -type f 2>/dev/null | wc -l || echo 0)
    json_count2=$(find ./src/gswarm-standalone-scheduler/scheduler_test_v2 -name "*.json" -type f 2>/dev/null | wc -l || echo 0)
    total_json=$((json_count + json_count2))
    if [ $total_json -gt 0 ]; then
        print_warning "Would remove $total_json test output JSON files"
    else
        print_success "No test output JSON files found"
    fi
fi

# Summary
print_header "Cleanup Complete"
if [ "$DRY_RUN" == "true" ]; then
    echo "This was a dry run. To actually delete files, run without --dry-run flag:"
    echo "  ./cleanup_debug.sh"
else
    print_success "All debug files have been cleaned up!"
fi

# Optionally show disk space saved
if command -v du &> /dev/null && [ "$DRY_RUN" == "false" ]; then
    echo -e "\n${GREEN}Tip: Run 'git gc' to clean up git objects and save more space${NC}"
fi
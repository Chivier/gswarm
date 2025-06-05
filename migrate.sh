#!/bin/bash

# Migration script for gswarm

echo "Starting migration to unified gswarm structure..."

# Create new directory structure
mkdir -p src/gswarm/{host,client,profiler,model,data,queue,utils}

# Copy Python files from old structure
echo "Copying files..."

# Copy profiler files
cp src/gswarm_profiler/*.py src/gswarm/profiler/ 2>/dev/null
cp src/gswarm_profiler/*.proto src/gswarm/profiler/ 2>/dev/null

# Copy model files  
cp src/gswarm_model/*.py src/gswarm/model/ 2>/dev/null
cp src/gswarm_model/*.proto src/gswarm/model/ 2>/dev/null

echo "Updating imports..."

# Update imports in profiler files
find src/gswarm/profiler -name "*.py" -exec sed -i 's/from \./from gswarm.profiler./g' {} \;
find src/gswarm/profiler -name "*.py" -exec sed -i 's/from gswarm_profiler/from gswarm.profiler/g' {} \;

# Update imports in model files
find src/gswarm/model -name "*.py" -exec sed -i 's/from \./from gswarm.model./g' {} \;
find src/gswarm/model -name "*.py" -exec sed -i 's/from gswarm_model/from gswarm.model/g' {} \;

echo "Migration complete!"
echo ""
echo "Next steps:"
echo "1. Review the migrated files in src/gswarm/"
echo "2. Test the new CLI with: python -m gswarm --help"
echo "3. Update any custom scripts to use the new CLI structure"
echo "4. Remove old directories when ready: rm -rf src/gswarm_profiler src/gswarm_model" 
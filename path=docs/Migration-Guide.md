# Migration Guide: From gswarm-profiler/gswarm-model to Unified gswarm

This guide helps you migrate from the separate `gswarm-profiler` and `gswarm-model` tools to the unified `gswarm` platform.

## Overview of Changes

### Unified CLI

The separate CLIs have been merged:
- `gsprof` → `gswarm profiler`
- `gsmodel` → `gswarm model`

### New Features

1. **Message Queue System**: Asynchronous task execution on clients
2. **Data Pool**: Distributed data management for model I/O
3. **Unified Configuration**: Single config file for all components
4. **Consistent Node Naming**: Same node names across all subsystems

### Port Changes

No changes to default ports:
- Profiler gRPC: 8090
- Profiler HTTP: 8091
- Model Host API: 9010
- Model Client API: 9011

## Migration Steps

### 1. Backup Existing Data

```bash
# Backup profiler data
cp -r ~/.gswarm_profiler_data ~/.gswarm_profiler_data.backup

# Backup model data
cp -r ~/.gswarm_model_data ~/.gswarm_model_data.backup
```

### 2. Uninstall Old Packages

```bash
pip uninstall gswarm-profiler gswarm-model
```

### 3. Install Unified gswarm

```bash
git clone https://github.com/yourusername/gswarm.git
cd gswarm
pip install .
```

### 4. Migrate Configuration

Create new unified config at `~/.gswarm/config.yaml`:

```yaml
# Merge your existing configs
cluster:
  host: "your-host-ip"
  port: 8090
  
profiling:
  # From gswarm-profiler config
  default_frequency: 1000
  enable_bandwidth: true
  
models:
  # From gswarm-model config
  storage_path: "/data/models"
  host_port: 9010
  client_port: 9011
  
# Add new queue config
queue:
  max_concurrent_tasks: 4
  
# Unified node list
nodes:
  - name: "node1"
    address: "192.168.1.101"
    # ... node config
```

### 5. Update Scripts

#### Profiler Commands

Old:
```bash
gsprof start --port 8090
gsprof connect host:8090
gsprof profile host:8090 --name test
```

New:
```bash
gswarm host start --port 8090
gswarm client connect host:8090
gswarm profiler start --name test
```

#### Model Commands

Old:
```bash
gsmodel host
gsmodel client --host host:9010
gsmodel download llama-7b
```

New:
```bash
gswarm host start  # Starts both profiler and model services
gswarm client connect host:8090
gswarm model download llama-7b
```

### 6. API Updates

#### Profiler APIs

No changes - all existing profiler APIs remain the same.

#### Model APIs

Updated to use `/api/v1/` prefix:

Old:
```bash
GET /models
POST /models/download/llama-7b
```

New:
```bash
GET /api/v1/models
POST /api/v1/models/llama-7b/download
```

### 7. Data Migration

The system will automatically detect and migrate existing data:

```bash
# Start unified host - it will migrate data
gswarm host start

# Check migration status
gswarm status --check-migration
```

## New Features Usage

### Message Queue

Take advantage of the new queue system:

```bash
# Submit multiple tasks - they'll queue automatically
gswarm model download model1 --priority high
gswarm model download model2 --priority normal
gswarm model download model3 --priority normal

# Check queue status
gswarm queue status
```

### Data Pool

Use the data pool for model chaining:

```bash
# Create data chunk
gswarm data create --source s3://bucket/data

# Use in model inference
gswarm model infer llama-7b --input-data chunk-123

# Output automatically goes to data pool
# Transfer to next node
gswarm data transfer <output-chunk> --to node2:dram
```

### Enhanced Move Operations

Model moves now have detailed status tracking:

```bash
# Move with progress tracking
gswarm model move llama-7b --from disk --to gpu0

# Check move status
gswarm model status llama-7b
# Shows: queued → moving → verifying → ready
```

## Rollback Plan

If you need to rollback:

1. Uninstall unified gswarm:
```bash
pip uninstall gswarm
```

2. Reinstall separate packages:
```bash
pip install gswarm-profiler gswarm-model
```

3. Restore backups:
```bash
mv ~/.gswarm_profiler_data.backup ~/.gswarm_profiler_data
mv ~/.gswarm_model_data.backup ~/.gswarm_model_data
```

## Breaking Changes

### CLI Changes

- All commands now under `gswarm` umbrella
- Some command names changed for consistency
- New required parameters for some operations

### API Changes

- Model APIs now use `/api/v1/` prefix
- Some response formats updated for consistency
- New error codes and formats

### Configuration

- Single unified config file
- Some parameter names changed
- New required sections

## Getting Help

- Check `gswarm --help` for command reference
- View logs in `~/.gswarm/logs/`
- Report issues on GitHub

## FAQ

**Q: Can I run old and new versions simultaneously?**
A: No, they use the same ports and data directories.

**Q: Will my profiling data be preserved?**
A: Yes, all existing data is automatically migrated.

**Q: Do I need to update all nodes at once?**
A: Yes, all nodes should run the same version.

**Q: Can I still use the HTTP APIs directly?**
A: Yes, the HTTP APIs are backward compatible with `/api/v1/` prefix. 
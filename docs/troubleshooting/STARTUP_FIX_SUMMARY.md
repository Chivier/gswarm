# GSwarm Startup Fix Summary

## Issues Fixed

### 1. AsyncIO Runtime Error
**Problem**: `RuntimeError: no running event loop` when importing the module

**Root Cause**: The `HeadState.__init__` method was trying to create an async task during module import, when no event loop was running.

**Solution**: 
- Removed `asyncio.create_task()` from `HeadState.__init__`
- Added `discovery_completed` flag to track state
- Moved model discovery to the FastAPI startup event

### 2. CLI Parameter Override
**Problem**: CLI parameters (--port, --http-port, --model-port) were not overriding `.gswarm.conf` settings

**Solution**:
- Modified `create_app()` function to accept CLI parameters
- Added config override logic in the app factory
- Updated host CLI to pass parameters to `create_app()`

## Changes Made

### 1. `src/gswarm/model/fastapi_head.py`

#### HeadState Class Changes
```python
# Before
def __init__(self):
    # ... other init code ...
    if config.host.auto_discover_models:
        asyncio.create_task(self.discover_and_register_models())  # ❌ Error!

# After  
def __init__(self):
    # ... other init code ...
    self.discovery_completed = False  # ✅ Safe flag
```

#### Startup Event Changes
```python
@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup"""
    # ... logging ...
    
    # Start model discovery if enabled and not already completed
    if config.host.auto_discover_models and not state.discovery_completed:
        logger.info("Starting model discovery...")
        await state.discover_and_register_models()
        state.discovery_completed = True
```

#### App Factory Changes
```python
def create_app(host: Optional[str] = None, port: Optional[int] = None, 
               model_port: Optional[int] = None, **kwargs) -> FastAPI:
    """Factory function with config overrides"""
    global config
    
    # Override config with CLI parameters if provided
    if host is not None or port is not None or model_port is not None:
        logger.info("Applying CLI parameter overrides...")
        # ... override logic ...
    
    return app
```

### 2. `src/gswarm/host/cli.py`

#### CLI Integration
```python
# Pass CLI parameters to create_app
model_app = create_app(host=host, port=port, model_port=model_port)
```

## Testing

Created test scripts to verify the fixes:

### 1. `test_startup_fix.py`
- Tests module import without asyncio errors
- Tests HeadState initialization
- Tests app creation with/without CLI overrides
- Tests configuration override functionality
- Tests manual discovery

### 2. `test_cli_command.py`
- Tests actual CLI command execution
- Verifies no startup errors
- Tests API connectivity with custom ports
- Verifies parameter override effectiveness

## Usage

### Basic Usage (Config File Only)
```bash
gswarm host start
```
Uses settings from `~/.gswarm.conf`

### CLI Parameter Override
```bash
gswarm host start --port 8095 --http-port 8096 --model-port 9010
```
Overrides config file with specified values

### Parameter Mapping
- `--port` → gRPC profiler port
- `--http-port` → HTTP API port  
- `--model-port` → Model management API port
- `--host` → Bind address

## Verification

To verify the fix works:

1. **Test Import Safety**:
   ```bash
   python test_startup_fix.py
   ```

2. **Test CLI Command**:
   ```bash
   python test_cli_command.py
   ```

3. **Test Actual Command**:
   ```bash
   gswarm host start --port 8095 --http-port 8096 --model-port 9010
   ```

## Benefits

1. **No More AsyncIO Errors**: Module imports safely without running event loop
2. **CLI Parameter Override**: Command line args override config file settings
3. **Backward Compatibility**: Existing behavior unchanged when no CLI args provided
4. **Proper Discovery**: Model discovery happens during startup event (async context)
5. **Flexible Configuration**: Mix of config file + CLI overrides supported

## Configuration Precedence

1. **CLI Parameters** (highest priority)
2. **Configuration File** (`~/.gswarm.conf`)
3. **Built-in Defaults** (lowest priority)

Example:
```yaml
# ~/.gswarm.conf
host:
  port: 8090        # Will be overridden by --port 8095
  model_port: 9000  # Will be overridden by --model-port 9010
  host: "0.0.0.0"   # Will be kept (no CLI override)
```

Running: `gswarm host start --port 8095 --model-port 9010`

Result:
- port: 8095 (from CLI)
- model_port: 9010 (from CLI)  
- host: "0.0.0.0" (from config file) 
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

gswarm-profiler is a distributed GPU cluster management system that combines:
- GPU profiling and monitoring across multiple nodes
- Model storage, deployment, and serving for ML workloads
- Data pooling for efficient cross-node data management
- Task orchestration with queue-based execution

The system uses a host-client architecture where a central host node coordinates operations across multiple client nodes.

## Common Development Commands

### Setup and Installation
```bash
# Install in development mode with all dependencies
pip install -e ".[dev]"

# Install with optional ML framework support
pip install -e ".[vllm]"      # For vLLM support
pip install -e ".[diffusion]"  # For diffusion models support
```

### Code Quality
```bash
# Format code (line length: 120)
black src/ tests/

# Lint code
ruff check src/ tests/
ruff check --fix src/ tests/  # Auto-fix issues

# Type checking
mypy src/

# Run pre-commit hooks on all files
pre-commit run --all-files
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_profiler.py

# Run with coverage
pytest --cov=src/gswarm

# Test model serving
./src/test/serve_and_test.sh [model_name] [device] [port]
```

### Running the Application
```bash
# Start host node
gswarm host start --port 8090 --http-port 8091 --model-port 9010

# Connect client node
gswarm client connect <host-ip>:8090 --resilient

# Start profiler
gswarm profiler start --name <session-name>

# Model operations
gswarm model list
gswarm model download <model-name> --source <source>
gswarm model serve <model-name> --device <device> --port <port>
```

## Code Architecture

### Core Components

1. **Host Node** (`src/gswarm/host/`)
   - Central coordinator managing model registry and task orchestration
   - Provides gRPC and HTTP APIs for client communication
   - Manages model distribution and task scheduling

2. **Client Node** (`src/gswarm/client/`)
   - Worker nodes that execute tasks and serve models
   - Connects to host for coordination
   - Handles local GPU resource management

3. **Profiler** (`src/gswarm/profiler/`)
   - GPU monitoring using nvitop
   - Streams metrics via gRPC
   - Supports multi-node profiling sessions

4. **Model Management** (`src/gswarm/model/`)
   - Distributed model storage with deduplication
   - Model serving with multiple framework support (vLLM, Transformers, Diffusers)
   - Version control and metadata management

5. **Data Pool** (`src/gswarm/data/`)
   - Cross-node data sharing with reference counting
   - Efficient data transfer and caching
   - Supports various data types and formats

6. **Queue System** (`src/gswarm/queue/`)
   - Task scheduling with dependency management
   - Priority-based execution
   - Distributed task execution across nodes

### Standalone Scheduler

The `src/gswarm-standalone-sheduler/` directory contains experimental scheduling implementations:
- **scheduler_test_v1**: Implements offline and online scheduling strategies
- Focuses on batching tasks by model type to minimize GPU switching overhead
- Uses YAML/JSON for workflow and request configurations

### Communication Patterns

- **gRPC**: High-performance metric streaming and control plane
- **HTTP/REST**: Model management and general APIs via FastAPI
- **Protocol Buffers**: Defined in `src/gswarm/profiler/proto/`

### Configuration

- Main config: `config.yaml` (see `config.yaml.example`)
- Project config: `pyproject.toml` (dependencies, tools, metadata)
- Python 3.9+ required
- Uses modern Python packaging with hatchling

### Key Development Practices

1. **Async-First**: Heavy use of asyncio for concurrent operations
2. **Type Safety**: Strict mypy checking enabled
3. **Code Style**: Black formatting with 120 char line length
4. **Testing**: pytest with async support and coverage reporting
5. **Excluded from linting**: Generated protobuf files (`*_pb2.py`, `*_pb2_grpc.py`)

### Current Development Focus

Based on git status, active development is in the standalone scheduler component:
- Modified: `offline_execution_log.json`, `offline_scheduler.py`
- New files: `__init__.py`, `scheduler_component.py`

Note: `offline_scheduler.py` imports a missing module `scheduler_queue` which needs to be resolved.
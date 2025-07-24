# Installation Guide

This guide provides step-by-step instructions for installing GSwarm and its dependencies.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Python**: Version 3.9 or higher.
- **Git**: For cloning the repository.
- **NVIDIA Drivers**: If you plan to use GSwarm with NVIDIA GPUs, make sure you have the appropriate drivers installed.

## Installation Steps

### 1. Clone the Repository

First, clone the GSwarm repository from GitHub:

```bash
git clone https://github.com/path/to/gswarm-profiler.git
cd gswarm-profiler
```

*(Note: Replace `https://github.com/path/to/gswarm-profiler.git` with the actual repository URL.)*

### 2. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies and avoid conflicts with other Python projects.

Create a virtual environment using `venv`:

```bash
python3 -m venv .venv
```

Activate the virtual environment:

- **On macOS and Linux:**
  ```bash
  source .venv/bin/activate
  ```
- **On Windows:**
  ```bash
  .venv\Scripts\activate
  ```

### 3. Install Dependencies

You can install GSwarm using `pip`. There are a few options depending on your use case.

#### Standard Installation

For a standard installation with core dependencies, run:

```bash
pip install .
```

#### Development Installation

If you plan to contribute to GSwarm or need the development tools, install it in editable mode (`-e`) with the `dev` extras. This will also install tools like `pytest`, `black`, and `ruff`.

```bash
pip install -e .[dev]
```

#### Full Installation (with all features)

To install GSwarm with all optional features, including support for `vLLM` and `diffusers`, use:

```bash
pip install -e .[dev,vllm,diffusion]
```

### 4. Verify Installation

After the installation is complete, you can verify it by running the `gswarm` command-line interface:

```bash
gswarm --help
```

This should display the main help menu with a list of available commands, confirming that the installation was successful.

## Configuration

GSwarm can be configured using a configuration file. An example configuration file is provided in the root of the repository:

- `gswarm.conf.example`

You can copy this file to your home directory as `.gswarm.conf` and customize it to your needs:

```bash
cp .gswarm.conf.example ~/.gswarm.conf
```

Alternatively, you can use CLI options to override settings from the configuration file. Refer to the documentation for each command for available options.

You are now ready to use GSwarm! For next steps, please refer to the [**Quick Start Guide**](Quick-Start.md).

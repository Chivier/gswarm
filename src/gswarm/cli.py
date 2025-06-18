"""
Unified CLI for gswarm - Distributed GPU cluster management system
"""

# Import subcommands
from .profiler import cli as profiler_cli
from .model import cli as model_cli
from .data import cli as data_cli
from .queue import cli as queue_cli
from .host import cli as host_cli
from .client import cli as client_cli

import typer
from typing import Optional
from loguru import logger
import sys

# Configure Loguru
logger.remove()
logger.add(sys.stderr, level="INFO")

# Create the main app with help text
app = typer.Typer(
    name="gswarm",
    help="Distributed GPU cluster management system with profiling and model orchestration",
    rich_markup_mode="rich",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)

# Add subcommands
app.add_typer(host_cli.app, name="host", help="Host node management")
app.add_typer(client_cli.app, name="client", help="Client node management")
app.add_typer(profiler_cli.app, name="profiler", help="GPU profiling operations")
app.add_typer(model_cli.app, name="model", help="Model management operations")
app.add_typer(data_cli.app, name="data", help="Data pool management")
app.add_typer(queue_cli.app, name="queue", help="Task queue management")


# Global callback to handle --yaml parameter
@app.callback()
def main_callback(
    yaml_config: Optional[str] = typer.Option(
        None, "--yaml", help="Path to YAML configuration file (overrides ~/.gswarm.conf)"
    ),
):
    """
    Gswarm - Distributed GPU cluster management system

    Use --yaml to specify a custom configuration file instead of ~/.gswarm.conf
    """
    if yaml_config:
        from .utils.config import set_config_path

        set_config_path(yaml_config)


# Add top-level commands
@app.command()
def status():
    """Get overall system status"""
    logger.info("Getting system status...")
    # TODO: Implement system-wide status check
    logger.info("System status check not yet implemented")


@app.command()
def health():
    """Check system health"""
    logger.info("Checking system health...")
    # TODO: Implement health check
    logger.info("Health check not yet implemented")


@app.command()
def nodes():
    """List all connected nodes"""
    logger.info("Listing connected nodes...")
    # TODO: Implement node listing
    logger.info("Node listing not yet implemented")


@app.command()
def version():
    """Show version information"""
    from . import __version__

    typer.echo(f"gswarm version: {__version__}")


@app.command(name="clean-history")
def clean_history():
    """Clean the gswarm cache directory"""
    from .utils.cache import clean_history

    if clean_history():
        typer.echo("✓ Cache directory cleaned successfully")
    else:
        typer.echo("✗ Failed to clean cache directory", err=True)
        raise typer.Exit(1)


def main():
    """Main entry point"""
    app()


if __name__ == "__main__":
    main()

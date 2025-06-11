"""Task queue CLI commands"""

import typer
from typing import Optional
from loguru import logger
import requests
from datetime import datetime

app = typer.Typer(help="Task queue management operations")


def get_api_url(host: str = "localhost:9011") -> str:
    """Ensure host has http:// prefix"""
    if not host.startswith("http://") and not host.startswith("https://"):
        return f"http://{host}"
    return host


@app.command()
def status(
    host: str = typer.Option("localhost:9011", "--host", help="Client API address"),
):
    """Get queue status"""
    try:
        url = f"{get_api_url(host)}/api/v1/queue"
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        logger.info("Queue Status:")
        logger.info(f"  Pending tasks: {data.get('pending', 0)}")
        logger.info(f"  Running tasks: {data.get('running', 0)}")
        logger.info(f"  Completed tasks: {data.get('completed', 0)}")

        if data.get("config"):
            config = data["config"]
            logger.info("\nQueue Configuration:")
            logger.info(f"  Max concurrent tasks: {config.get('max_concurrent_tasks', 'N/A')}")
            logger.info(f"  Priority levels: {', '.join(config.get('priority_levels', []))}")
            logger.info(f"  Resource tracking: {'enabled' if config.get('resource_tracking') else 'disabled'}")
    except Exception as e:
        logger.error(f"Failed to get queue status: {e}")


@app.command()
def tasks(
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status"),
    limit: int = typer.Option(20, "--limit", "-l", help="Number of tasks to show"),
    host: str = typer.Option("localhost:9011", "--host", help="Client API address"),
):
    """List tasks in the queue"""
    try:
        url = f"{get_api_url(host)}/api/v1/queue/tasks"
        params = {"limit": limit}
        if status:
            params["status"] = status

        response = requests.get(url, params=params)
        response.raise_for_status()

        tasks = response.json().get("tasks", [])

        if tasks:
            logger.info(f"Found {len(tasks)} task(s):")
            for task in tasks:
                logger.info(f"\n  Task ID: {task['task_id']}")
                logger.info(f"    Type: {task['task_type']}")
                logger.info(f"    Status: {task['status']}")
                logger.info(f"    Priority: {task['priority']}")

                if task.get("created_at"):
                    created = datetime.fromtimestamp(task["created_at"])
                    logger.info(f"    Created: {created.strftime('%Y-%m-%d %H:%M:%S')}")

                if task.get("started_at"):
                    started = datetime.fromtimestamp(task["started_at"])
                    logger.info(f"    Started: {started.strftime('%Y-%m-%d %H:%M:%S')}")

                if task.get("completed_at"):
                    completed = datetime.fromtimestamp(task["completed_at"])
                    duration = task["completed_at"] - task.get("started_at", task["created_at"])
                    logger.info(f"    Completed: {completed.strftime('%Y-%m-%d %H:%M:%S')} (took {duration:.1f}s)")

                if task.get("dependencies"):
                    logger.info(f"    Dependencies: {', '.join(task['dependencies'])}")

                if task.get("resources"):
                    res = task["resources"]
                    if res.get("devices"):
                        logger.info(f"    Devices: {', '.join(res['devices'])}")
                    if res.get("models"):
                        logger.info(f"    Models: {', '.join(res['models'])}")
        else:
            logger.info("No tasks found")
    except Exception as e:
        logger.error(f"Failed to list tasks: {e}")


@app.command()
def cancel(
    task_id: str = typer.Argument(..., help="Task ID to cancel"),
    host: str = typer.Option("localhost:9011", "--host", help="Client API address"),
):
    """Cancel a pending or running task"""
    try:
        url = f"{get_api_url(host)}/api/v1/queue/tasks/{task_id}/cancel"
        response = requests.post(url)
        response.raise_for_status()

        result = response.json()
        if result.get("success"):
            logger.info(f"Task '{task_id}' cancelled successfully")
            if result.get("message"):
                logger.info(f"  {result['message']}")
        else:
            logger.error(f"Failed to cancel task: {result.get('message', 'Unknown error')}")
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            logger.error(f"Task '{task_id}' not found")
        else:
            logger.error(f"Failed to cancel task: {e}")
    except Exception as e:
        logger.error(f"Failed to cancel task: {e}")


@app.command()
def history(
    limit: int = typer.Option(50, "--limit", "-l", help="Number of records to show"),
    since: Optional[str] = typer.Option(None, "--since", help="Show tasks since timestamp"),
    host: str = typer.Option("localhost:9011", "--host", help="Client API address"),
):
    """Get task execution history"""
    try:
        url = f"{get_api_url(host)}/api/v1/queue/history"
        params = {"limit": limit}
        if since:
            params["since"] = since

        response = requests.get(url, params=params)
        response.raise_for_status()

        history = response.json().get("history", [])

        if history:
            logger.info(f"Task History ({len(history)} records):")

            # Group by status
            by_status = {}
            for task in history:
                status = task.get("status", "unknown")
                if status not in by_status:
                    by_status[status] = []
                by_status[status].append(task)

            for status, tasks in by_status.items():
                logger.info(f"\n  {status.upper()} ({len(tasks)} tasks):")
                for task in tasks[:5]:  # Show first 5 of each status
                    logger.info(f"    - {task['task_id']} ({task['task_type']})")
                    if task.get("completed_at") and task.get("started_at"):
                        duration = task["completed_at"] - task["started_at"]
                        logger.info(f"      Duration: {duration:.1f}s")

                if len(tasks) > 5:
                    logger.info(f"    ... and {len(tasks) - 5} more")
        else:
            logger.info("No task history found")
    except Exception as e:
        logger.error(f"Failed to get task history: {e}")


@app.command()
def clear(
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Clear only tasks with this status"),
    force: bool = typer.Option(False, "--force", "-f", help="Force clear without confirmation"),
    host: str = typer.Option("localhost:9011", "--host", help="Client API address"),
):
    """Clear completed or failed tasks from history"""
    if not force:
        confirm = typer.confirm(f"Clear {'all' if not status else status} tasks from history?")
        if not confirm:
            logger.info("Operation cancelled")
            return

    try:
        url = f"{get_api_url(host)}/api/v1/queue/clear"
        data = {}
        if status:
            data["status"] = status

        response = requests.post(url, json=data)
        response.raise_for_status()

        result = response.json()
        logger.info(f"Cleared {result.get('cleared', 0)} tasks from history")
    except Exception as e:
        logger.error(f"Failed to clear task history: {e}")

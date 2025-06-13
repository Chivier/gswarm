import os
import sys
import tempfile
from loguru import logger
import time
import uuid


def get_pid_file(component: str = "client") -> str:
    """
    Get PID file path based on whether it's a host or client.
    If host is True, returns the PID file for the host process.
    If host is False, returns the PID file for the client process.
    The PID file is stored in the system's temporary directory.
    :param host: Boolean indicating if the PID file is for the host process.
    :return: Path to the PID file.
    """
    return os.path.join(tempfile.gettempdir(), f"gswarm_{component}.pid")


def get_log_filepath(component: str = "client") -> str:
    timestamp = int(time.time())
    unique_id = uuid.uuid4().hex[:8]

    if not os.path.exists(f"/tmp/gswarm_{unique_id}"):
        os.makedirs(f"/tmp/gswarm_{unique_id}")

    log_file_path = f"/tmp/gswarm_{unique_id}/gswarm_{component}_{timestamp}.log"
    return log_file_path


def check_pid_file_exists(pid_file_path: str) -> bool:
    """
    Check if the PID file exists.
    :param pid_file_path: Path to the PID file.
    :return: True if the PID file exists, False otherwise.
    """
    return os.path.exists(pid_file_path)


def daemonize(log_file_path: str = None):
    try:
        pid = os.fork()
        if pid > 0:
            # exit parent process
            sys.exit(0)
    except OSError as e:
        sys.stderr.write(f"fork #1 failed: {e}\n")
        sys.exit(1)

    # detach from parent environment
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.setsid()
    os.umask(0)

    # second fork
    try:
        pid = os.fork()
        if pid > 0:
            sys.exit(0)
    except OSError as e:
        sys.stderr.write(f"fork #2 failed: {e}\n")
        sys.exit(1)

    sys.stdout.flush()
    sys.stderr.flush()
    with open("/dev/null", "rb", 0) as f:
        os.dup2(f.fileno(), sys.stdin.fileno())
    with open(log_file_path, "ab", 0) as f:
        os.dup2(f.fileno(), sys.stdout.fileno())
    with open(log_file_path, "ab", 0) as f:
        os.dup2(f.fileno(), sys.stderr.fileno())

    logger.info(f"Daemon started with PID {os.getpid()}, logs redirected to {log_file_path}")

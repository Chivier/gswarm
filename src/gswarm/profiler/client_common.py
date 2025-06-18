import importlib.util
import ast
from loguru import logger


def find_function_names(file_path):
    """
    Use ast static analysis to find all top-level function names in a Python file.

    :param file_path: Path to the target Python file.
    :return: A list containing all function names.
    """
    with open(file_path, "r", encoding="utf-8") as source:
        tree = ast.parse(source.read())

    function_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    return function_names


def dynamic_import_metrics(file_path, function_names):
    """
    Dynamically import a module and get functions with specified names from it.

    :param file_path: Path to the target Python file.
    :param function_names: List of function names to import.
    :return: A dictionary containing specified function names and function objects.
    """
    spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    functions = {name[4:-8]: getattr(module, name) for name in function_names if hasattr(module, name)}
    return functions


def parse_extra_metrics():
    function_names = find_function_names("src/gswarm/profiler/extra_metrics.py")

    available_metrics = [name for name in function_names if name.startswith("get_") and name.endswith("_metrics")]
    available_metrics = dynamic_import_metrics("src/gswarm/profiler/extra_metrics.py", available_metrics)

    logger.info(f"Enabled extra metrics: {list(available_metrics.keys())}")
    return available_metrics

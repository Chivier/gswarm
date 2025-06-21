import importlib.util
import ast
from loguru import logger
import nvitop


def check_nvitop_support(deivce, name: str):
    if not hasattr(deivce, name):
        return False
    value = getattr(deivce, name)()
    if value is None or value == nvitop.NA:
        return False
    return True


def parse_extra_metrics(metrics_list: list[str]) -> list[str]:
    """
    Check support metrics in metric list and return supported metrics.
    """

    test_device = nvitop.Device(0)  # By default, we use device 0 to check support.
    supported_metrics = []
    for metric in metrics_list:
        if hasattr(test_device, metric):
            value = getattr(test_device, metric)()
            if value is not None and value != nvitop.NA:
                supported_metrics.append(metric)
            else:
                logger.warning(f"Metric '{metric}' is not supported by the device.")
        else:
            logger.warning(f"Metric '{metric}' does not exist in the device.")
    return supported_metrics


def get_extra_metrics_value(device, metrics: list[str]) -> dict[str, float]:
    """
    Get the value of extra metrics from the device.
    """
    metrics_value = {}
    for metric in metrics:
        value = getattr(device, metric)()
        metrics_value[metric] = value if value is not None and value != nvitop.NA else 0.0

    return metrics_value

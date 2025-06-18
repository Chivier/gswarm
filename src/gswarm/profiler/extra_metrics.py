import nvitop


def get_gpu_power_metrics(device):
    """
    Get GPU power metrics for a given device.

    Args:
        device (nvitop.Device): The nvitop device object.

    Returns:
        dict: A dictionary containing the GPU power metrics.
    """
    power_info = {}
    power_draw = device.power_draw()
    power_limit = device.power_limit()
    power_usage = device.power_usage()

    if power_draw is None or power_draw == nvitop.NA:
        power_info["power_draw"] = None
    else:
        power_info["power_draw"] = power_draw

    if power_limit is None or power_limit == nvitop.NA:
        power_info["power_limit"] = None
    else:
        power_info["power_limit"] = power_limit

    if power_usage is None or power_usage == nvitop.NA:
        power_info["power_usage"] = None
    else:
        power_info["power_usage"] = power_usage

    return power_info

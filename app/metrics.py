import time

_metrics = {}


def start_timer(name: str):
    """
    Start a timer for a pipeline step.
    """
    _metrics[name] = {
        "start": time.perf_counter(),
        "duration": None,
    }


def stop_timer(name: str):
    """
    Stop the timer and store duration.
    """
    if name in _metrics:
        _metrics[name]["duration"] = (
            time.perf_counter() - _metrics[name]["start"]
        )


def get_metrics():
    """
    Return pipeline timings.
    """
    return {
        key: value["duration"]
        for key, value in _metrics.items()
    }


def reset_metrics():
    """
    Clear previous timings.
    """
    _metrics.clear()

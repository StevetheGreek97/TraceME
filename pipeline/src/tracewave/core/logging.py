import contextvars
import logging
import os
import socket
import time
from contextlib import contextmanager

# Optional CUDA/NVML snapshots (works with your existing nvidia-ml-py 13.x)
try:
    import torch
except Exception:
    torch = None

try:
    from pynvml import (
        nvmlInit, nvmlShutdown, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo,
    )
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

_DEF_FMT = "%(asctime)s | %(levelname)-7s | %(run_id)s | job=%(job_id)s | %(name)s | pid=%(pid)s host=%(host)s | %(message)s"
_DEF_DATE = "%H:%M:%S"

_LOG_CONTEXT = contextvars.ContextVar("log_context", default={})
_HOSTNAME = socket.gethostname()
_LOG_CONFIGURED = False


def _level_from_env(level: str | int | None) -> int:
    lvl = os.getenv("LOGLEVEL", None)
    if lvl is not None:
        level = lvl
    if isinstance(level, str):
        return getattr(logging, level.upper(), logging.INFO)
    if isinstance(level, int):
        return level
    return logging.INFO


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        ctx = _LOG_CONTEXT.get()
        record.run_id = ctx.get("run_id", "-")
        record.job_id = ctx.get("job_id", "-")
        record.host = _HOSTNAME
        record.pid = os.getpid()
        return True


def configure_logging(
    *,
    level: str | int | None = None,
    fmt: str | None = None,
    datefmt: str | None = None,
) -> None:
    global _LOG_CONFIGURED
    if _LOG_CONFIGURED:
        return

    root = logging.getLogger()
    if root.handlers:
        _LOG_CONFIGURED = True
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt or _DEF_FMT, datefmt=datefmt or _DEF_DATE))
    handler.addFilter(_ContextFilter())
    root.addHandler(handler)
    root.setLevel(_level_from_env(level))
    _LOG_CONFIGURED = True


def add_file_handler(log_file: os.PathLike | str, *, level: str | int | None = None) -> None:
    configure_logging(level=level)
    root = logging.getLogger()
    fh = logging.FileHandler(log_file)
    fh.setLevel(_level_from_env(level))
    fh.setFormatter(logging.Formatter(_DEF_FMT, datefmt=_DEF_DATE))
    fh.addFilter(_ContextFilter())
    root.addHandler(fh)


def set_log_context(**kwargs) -> None:
    ctx = dict(_LOG_CONTEXT.get())
    ctx.update({k: v for k, v in kwargs.items() if v is not None})
    _LOG_CONTEXT.set(ctx)


def clear_log_context() -> None:
    _LOG_CONTEXT.set({})


def get_logger(name: str = "tracewave", level: str | int | None = None) -> logging.Logger:
    """
    Get a configured logger (idempotent). Uses root handlers.
    """
    configure_logging(level=level)
    log = logging.getLogger(name)
    if level is not None:
        log.setLevel(_level_from_env(level))
    return log

@contextmanager
def timer(log: logging.Logger, label: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        log.info(f"{label} finished in {dt:.3f}s")

def _cuda_mem_str() -> str:
    if torch is None or not torch.cuda.is_available():
        return "CUDA not available"
    try:
        idx = torch.cuda.current_device()
        total, free = torch.cuda.mem_get_info(idx)
        used = total - free
        return (f"GPU{idx} mem: used {used/1e9:.2f} GB / total {total/1e9:.2f} GB "
                f"(free {free/1e9:.2f} GB)")
    except Exception as e:
        return f"cuda.mem_get_info failed: {e}"

def _nvml_str() -> str:
    if not _HAS_NVML:
        return "NVML not available"
    try:
        nvmlInit()
        n = nvmlDeviceGetCount()
        parts = []
        for i in range(n):
            h = nvmlDeviceGetHandleByIndex(i)
            util = nvmlDeviceGetUtilizationRates(h)
            mem = nvmlDeviceGetMemoryInfo(h)
            parts.append(
                f"GPU{i} util {util.gpu:3d}% | mem {mem.used/1e9:.2f}/{mem.total/1e9:.2f} GB"
            )
        return " | ".join(parts)
    except Exception as e:
        return f"NVML snapshot failed: {e}"
    finally:
        try:
            nvmlShutdown()
        except Exception:
            pass

def log_gpu_snapshot(log: logging.Logger, tag: str = ""):
    """
    One-line snapshot combining torch and NVML (if available).
    """
    tagp = f"[{tag}] " if tag else ""
    cuda_line = _cuda_mem_str()
    nvml_line = _nvml_str()
    log.info(f"{tagp}{cuda_line} || {nvml_line}")

from __future__ import annotations
import logging
import sys
from typing import Optional

# Custom verbosity levels
VERBOSITY_QUIET = 0      # Only errors
VERBOSITY_NORMAL = 1     # Info + warnings
VERBOSITY_VERBOSE = 2    # Debug messages
VERBOSITY_DEBUG = 3      # All debug + trace

_VERBOSITY: int = VERBOSITY_NORMAL
_LOGGER: Optional[logging.Logger] = None


class ColorFormatter(logging.Formatter):
    """Formatter with colors for terminal output."""

    COLORS = {
        logging.DEBUG: '\033[36m',     # Cyan
        logging.INFO: '\033[32m',      # Green
        logging.WARNING: '\033[33m',   # Yellow
        logging.ERROR: '\033[31m',     # Red
        logging.CRITICAL: '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def __init__(self, use_color: bool = True):
        super().__init__()
        self.use_color = use_color and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        # Customize format based on level
        if record.levelno == logging.DEBUG:
            fmt = "[DBG] %(message)s"
        elif record.levelno == logging.INFO:
            fmt = "[INFO] %(message)s"
        elif record.levelno == logging.WARNING:
            fmt = "[WARN] %(message)s"
        else:
            fmt = "[%(levelname)s] %(message)s"

        if self.use_color:
            color = self.COLORS.get(record.levelno, '')
            fmt = f"{color}{fmt}{self.RESET}"

        formatter = logging.Formatter(fmt)
        return formatter.format(record)


def set_verbosity(level: int) -> None:
    """
    Set global verbosity level.

    Args:
        level: 0=quiet, 1=normal, 2=verbose, 3=debug
    """
    global _VERBOSITY
    _VERBOSITY = level

    if _LOGGER:
        if level == VERBOSITY_QUIET:
            _LOGGER.setLevel(logging.ERROR)
        elif level == VERBOSITY_NORMAL:
            _LOGGER.setLevel(logging.INFO)
        elif level == VERBOSITY_VERBOSE:
            _LOGGER.setLevel(logging.DEBUG)
        else:
            _LOGGER.setLevel(logging.DEBUG)


def get_verbosity() -> int:
    """Return current verbosity level."""
    return _VERBOSITY


def get_logger(
    name: str = "grayshield",
    level: Optional[int] = None,
    use_color: bool = True,
) -> logging.Logger:
    """
    Get or create the GrayShield logger.

    Args:
        name: Logger name
        level: Logging level (if None, uses verbosity setting)
        use_color: Whether to use colored output

    Returns:
        Configured logger instance
    """
    global _LOGGER

    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColorFormatter(use_color=use_color))
        logger.addHandler(handler)
        logger.propagate = False

    # Set level based on verbosity or explicit level
    if level is not None:
        logger.setLevel(level)
    else:
        if _VERBOSITY == VERBOSITY_QUIET:
            logger.setLevel(logging.ERROR)
        elif _VERBOSITY == VERBOSITY_NORMAL:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.DEBUG)

    _LOGGER = logger
    return logger


def log_experiment_start(
    rq: str,
    model: str,
    task: str,
    payload_path: str,
    **kwargs,
) -> None:
    """Log experiment start with configuration details."""
    logger = get_logger()
    logger.info(f"{'='*60}")
    logger.info(f"Starting {rq}: {model} on {task}")
    logger.info(f"Payload: {payload_path}")

    if _VERBOSITY >= VERBOSITY_VERBOSE:
        for key, value in kwargs.items():
            logger.debug(f"  {key}: {value}")

    logger.info(f"{'='*60}")


def log_experiment_result(
    rq: str,
    metrics: dict,
    out_dir: str,
) -> None:
    """Log experiment results."""
    logger = get_logger()
    logger.info(f"--- {rq} Results ---")

    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

    logger.info(f"Results saved to: {out_dir}")


def log_defense_applied(
    defense_type: str,
    n_params: int,
    elapsed_ms: float,
) -> None:
    """Log defense application details."""
    logger = get_logger()
    logger.info(f"Applied {defense_type} defense to {n_params:,} parameters in {elapsed_ms:.1f}ms")


def log_timing(operation: str, elapsed_seconds: float) -> None:
    """Log timing information."""
    logger = get_logger()
    if _VERBOSITY >= VERBOSITY_VERBOSE:
        logger.debug(f"Timing: {operation} took {elapsed_seconds*1000:.1f}ms")

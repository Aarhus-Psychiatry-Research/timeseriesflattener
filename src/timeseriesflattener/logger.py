"""Exampel of how to setup a logger with sensible defaults."""

import logging
from pathlib import Path

import coloredlogs

from timeseriesflattener.utils import PROJECT_ROOT


def setup_logger(
    name: str,
    level: int = logging.DEBUG,
    log_file_path: str = None,
    format: str = "%(asctime)s [%(levelname)s] %(message)s",
) -> logging.Logger:
    """
    Set up a logger with the given name and log level.

    Args:
        name (str): The name of the logger. Typically use __name__.
        level (int): The log level of the logger. The default is None.
        log_file_path (str): The path to the log file where the logger should
            output log messages. If not specified, the logger does not output
            log messages to a file.
        format (str): The format of the log messages.

    Returns:
        The logger object.

    The logger outputs log messages to the console and to a log file, if a
    log file path is specified.
    """
    coloredlogs.install()

    # create logger
    logger = logging.getLogger(name)

    if level:
        logger.setLevel(level)

    # create console handler and set level to debug
    ch = logging.StreamHandler()

    if level:
        ch.setLevel(level)

    if format:
        # create formatter
        formatter = logging.Formatter(format)
        # add formatter to ch
        ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    # create a file handler and set level to debug
    if log_file_path:
        logfile_dir = Path(log_file_path).parent

        if not logfile_dir.exists():
            logfile_dir.mkdir(parents=True)

        fh = logging.FileHandler(log_file_path)

        if level:
            fh.setLevel(level)
        if format:
            fh.setFormatter(formatter)

        logger.addHandler(fh)

    return logger


if __name__ == "__main__":
    # This is an example of how to setup a root logger
    logger = setup_logger(
        "",
        level=logging.DEBUG,
        log_file_path=PROJECT_ROOT / "logs" / "test.log",
        format="%(asctime)s [%(levelname)s]: %(message)s",
    )

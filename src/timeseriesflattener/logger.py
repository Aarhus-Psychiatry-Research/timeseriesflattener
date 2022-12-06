"""Example of how to setup a logger with sensible defaults."""

import logging
from pathlib import Path

import coloredlogs

from timeseriesflattener.utils import PROJECT_ROOT


def setup_logger(
    name: str,
    level: int = logging.DEBUG,
    log_file_path: str = None,
    fmt: str = "%(asctime)s [%(levelname)s] %(message)s",
) -> logging.Logger:
    """
    Set up a logger with the given name and log level.

    Args:
        name (str): The name of the logger. Typically use __name__.
        level (int): The log level of the logger. The default is None.
        log_file_path (str): The path to the log file where the logger should
            output log messages. If not specified, the logger does not output
            log messages to a file.
        fmt (str): The fmt of the log messages.

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
    channel = logging.StreamHandler()

    if level:
        channel.setLevel(level)

    if fmt:
        # create formatter
        formatter = logging.Formatter(fmt)
        # add formatter to ch
        channel.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(channel)

    # create a file handler and set level to debug
    if log_file_path:
        logfile_dir = Path(log_file_path).parent

        if not logfile_dir.exists():
            logfile_dir.mkdir(parents=True)

        file_handler = logging.FileHandler(log_file_path)

        if level:
            file_handler.setLevel(level)
        if fmt:
            file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    # This is an example of how to setup a root logger
    logger = setup_logger(
        "",
        level=logging.DEBUG,
        log_file_path=PROJECT_ROOT / "logs" / "test.log",
        fmt="%(asctime)s [%(levelname)s]: %(message)s",
    )

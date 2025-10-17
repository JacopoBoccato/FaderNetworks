# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
import os
from datetime import timedelta


class LogFormatter:

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message)


def create_logger(filepath):
    """
    Create and configure a logger for console and (optionally) file output.
    Supports UTF-8 encoding for multilingual data (e.g., Arabic in MSA datasets).
    """
    log_formatter = LogFormatter()

    # create file handler if a path is provided
    if filepath is not None:
        # ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        file_handler = logging.FileHandler(filepath, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # configure root logger
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # utility to reset timing
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    return logger

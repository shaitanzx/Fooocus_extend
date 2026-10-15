import logging
from logging import handlers
from typing import Optional


class Logger:

    def __init__(self, level="INFO", log_file: Optional[str] = None):
        self.logger = logging.getLogger()
        self.logger.setLevel(level)

        formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        if log_file:
            file_handler = handlers.TimedRotatingFileHandler(filename=log_file,
                                                             when='D',
                                                             interval=1,
                                                             backupCount=5,
                                                             encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        else:
            self.logger.warning("save_log not enable or log file path not exist, log will only output in console.")

    def set_level(self, level):
        if level.lower() == "debug":
            level = logging.DEBUG
        elif level.lower() == "info":
            level = logging.INFO
        elif level.lower() == "warning":
            level = logging.WARNING
        elif level.lower() == "error":
            level = logging.ERROR
        elif level.lower() == "critical":
            level = logging.CRITICAL
        else:
            error_message = "Invalid log level"
            self.logger.critical(error_message)
            raise ValueError(error_message)

        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

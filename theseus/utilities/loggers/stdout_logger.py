import logging
from .observer import LoggerSubscriber

class CustomFormatter(logging.Formatter):
    """
    Color schemes longging formater
    https://docs.microsoft.com/en-us/windows/terminal/customize-settings/color-schemes
    """
    grey = "\x1b[38;21m"
    green = "\x1b[1;32m"
    yellow = "\x1b[33;21m"
    bold_red = "\x1b[31;21m"
    red = "\x1b[31;1m"
    grey2 = "\x1b[1;30m"
    white = "\x1b[1;37m"
    reset = "\x1b[0m"
    cyan = "\x1b[1;36m"
    purple = "\x1b[35m"

    FORMATS = {
        logging.DEBUG: green,
        logging.INFO: cyan,
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: bold_red
    }

    def __init__(self, text_format, date_format):
        self.text_format = text_format
        self.date_format = date_format

    def format(self, record):
        log_fmt = self.text_format.format(
            level_color=self.FORMATS.get(record.levelno),
            time_color=self.grey2, msg_color=self.white, 
            path_color=self.purple)

        formatter = logging.Formatter(log_fmt, datefmt=self.date_format)
        return formatter.format(record)


class StdoutLogger(LoggerSubscriber):
    """
    Logger class for showing text in prompt and file
    For more documents, look into https://docs.python.org/3/library/logging.html
    
    Usage:
        from modules.logger import StdoutLogger
        LOGGER = StdoutLogger.init_logger(__name__)

    """

    date_format = '%d-%m-%y %H:%M:%S'
    message_format = '[%(asctime)s][%(filename)s::%(lineno)d][%(levelname)s]: %(message)s'
    color_message_format = '{time_color}[%(asctime)s]\x1b[0m{path_color}[%(filename)s::%(lineno)d]\x1b[0m{level_color}[%(levelname)s]\x1b[0m: {msg_color}%(message)s\x1b[0m'
    def __init__(self, name, logdir, debug=False):
        self.logdir = logdir
        self.filename = f'{self.logdir}/log.txt'

        if debug:
            self.level = logging.DEBUG        
        else:
            self.level = logging.INFO        

        # Init logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)

        # Create handlers
        handlers = self.init_handlers()

        # Add handlers
        self.add_handlers(self.logger, handlers=handlers)

    def init_handlers(self):
        # Create one file logger and one stream logger
        stream_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(self.filename)
        
        # Create formatters and add it to handlers
        format = logging.Formatter(StdoutLogger.message_format, datefmt=StdoutLogger.date_format)
        custom_format = CustomFormatter(StdoutLogger.color_message_format, date_format=StdoutLogger.date_format)
        stream_handler.setFormatter(custom_format)
        file_handler.setFormatter(format)
        
        return stream_handler, file_handler

    def add_handlers(self, logger, handlers):
        # Add handlers to the logger
        for handler in handlers:
            logger.addHandler(handler)

    def set_debug_mode(self, toggle="off"):
        if toggle == "on":
            self.level = logging.DEBUG
        else:
            self.level = logging.INFO
        self.logger.setLevel(self.level)

    def log_text(self, tag, value, level, **kwargs):
        if level == logging.WARN:
            self.logger.warn(value)

        if level == logging.INFO:
            self.logger.info(value)

        if level == logging.ERROR:
            self.logger.error(value)

        if level == logging.DEBUG:
            self.logger.debug(value)
        
import logging
import logging.config
import os


def configure_logger(
    logger=None, cfg=None, log_file=None, console=True, log_level="DEBUG"
):
    """Function to setup configurations of logger through function.

    The individual arguments of `log_file`, `console`, `log_level` will overwrite the ones in cfg.

    Args:
            logger:
                    Predefined logger object if present. If None a ew logger object will be created from root.
            cfg: dict()
                    Configuration of the logging to be implemented by default
            log_file: str
                    Path to the log file for logs to be stored
            console: bool
                    To include a console handler(logs printing in console)
            log_level: str
                    One of `["INFO","DEBUG","WARNING","ERROR","CRITICAL"]`
                    default - `"DEBUG"`

    Returns:
        logging.Logger
    """
    logging.config.dictConfig(cfg)

    logger = logger or logging.getLogger()

    if log_file or console:
        for hdlr in logger.handlers:
            logger.removeHandler(hdlr)

        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(getattr(logging, log_level))
            logger.addHandler(fh)

        if console:
            sh = logging.StreamHandler()
            sh.setLevel(getattr(logging, log_level))
            logger.addHandler(sh)

    return logger


def logger_start(log_level, log_path, console_):
    # configuring and assigning in the logger can be done by the below function
    """Starts Logger for the program.

    Args:
        logger:
            Predefined logger object if present. If None a ew logger object will be created from root.
        cfg: dict()
            Configuration of the logging to be implemented by default
        log_file: str
            Path to the log file for logs to be stored
        console: bool
            To include a console handler(logs printing in console)
        log_level: str
            One of `["INFO","DEBUG","WARNING","ERROR","CRITICAL"]`
            default - `"DEBUG"`

    Returns:
        logging.Logger
    """
    LOGGING_DEFAULT_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "simple": {"format": "%(message)s"},
        },
        "root": {"level": log_level},
    }

    if not log_path:
        file = None
    else:
        os.makedirs(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), log_path),
            exist_ok=True,
        )
        file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), log_path, "config.log"
        )

    logger = configure_logger(
        log_file=file,
        cfg=LOGGING_DEFAULT_CONFIG,
        console=console_,
    )
    logger.info("Logger Started.")
    return logger

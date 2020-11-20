import datetime
import logging
from os import makedirs
from os.path import dirname, abspath, join, exists
import pytz


def get_root_path():
    return dirname(dirname(abspath(__file__)))


def get_logs_path():
    return join(get_root_path(), 'logs')


def get_data_path():
    return join(get_root_path(), 'data')


def get_temp_path():
    return join(get_root_path(), 'temp')


def create_dir_if_not_exists(folder):
    if not exists(folder):
        makedirs(folder)


def log_wrap(log_name, console=False, log_file=False, file_name=join(get_logs_path(), 'log.txt')):
    """
    logging module wrapper
    Parameters
    ----------
    log_name  : str
        name to use for creating logger
    console   : bool
        logging to console
    log_file  : bool
        logging to specified log file
    file_name : str
        name of the file to use for logging if log_file is set to True
    Returns
    -------
    logger : logger object
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '[%(asctime)s - %(name)s - %(funcName)s - '
        '%(levelname)s] - %(message)s')
    if console:  # add console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if log_file:  # add file handler
        fh = logging.FileHandler(file_name)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def get_current_ts(zone='US/Pacific'):
    return datetime.datetime.now(pytz.timezone(zone)).strftime(
        '%Y-%m-%dT%H-%M-%S.%f')
"""
This module sets up all logger configuration.

It reads all configs from enviromnent variables and a
.env file using Pydantic.
"""

from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggerSettings(BaseSettings):
    """Logger configuration settings for the application.

    Args:
        logger_config (SettingsConfigDict): logger configs from .env file.
        log_level (str): Logging level.
    """

    model_config = SettingsConfigDict(
        env_file='config/.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )
    model_config['protected_namespaces'] = ('settings_',)

    log_level: str


def configure_logging(log_level: str) -> None:
    """Configure the logger.

    Args:
        log_level (str): The log level to be set for the logger.
    """
    logger.remove()
    logger.add(
        'logs/app.log',
        backtrace=False,
        diagnose=False,
        rotation='1 day',  # Time required to start logging into a new file.
        compression='zip',  # Method used for compressing old log files automatically.
        level=log_level,
    )


# Configuring logging
configure_logging(log_level=LoggerSettings().log_level)

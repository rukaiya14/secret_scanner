"""
Structured JSON logging infrastructure for the ML-Enhanced Secret Scanner.

This module provides a centralized logging system with structured JSON output
for better observability and debugging in production environments.
"""

import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional
import uuid


class JSONFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs in structured JSON format.
    
    This enables better log parsing and analysis in production monitoring systems.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON-formatted log string
        """
        log_data = {
            "timestamp": datetime.now(datetime.UTC).isoformat().replace('+00:00', 'Z'),
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
        }
        
        # Add trace ID if available
        if hasattr(record, 'trace_id'):
            log_data["trace_id"] = record.trace_id
        
        # Add error code if available
        if hasattr(record, 'error_code'):
            log_data["error_code"] = record.error_code
        
        # Add context if available
        if hasattr(record, 'context'):
            log_data["context"] = record.context
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'levelno', 'lineno', 'module', 'msecs',
                          'message', 'pathname', 'process', 'processName',
                          'relativeCreated', 'thread', 'threadName', 'exc_info',
                          'exc_text', 'stack_info', 'trace_id', 'error_code', 'context']:
                log_data[key] = value
        
        return json.dumps(log_data)


class StructuredLogger:
    """
    Wrapper around Python logger that adds structured logging capabilities.
    
    Provides methods for logging with additional context, error codes, and trace IDs.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize structured logger.
        
        Args:
            logger: Python logger instance to wrap
        """
        self.logger = logger
        self.trace_id = None
    
    def set_trace_id(self, trace_id: Optional[str] = None) -> str:
        """
        Set trace ID for correlating related log entries.
        
        Args:
            trace_id: Trace ID to use (generates new UUID if None)
            
        Returns:
            The trace ID that was set
        """
        self.trace_id = trace_id or str(uuid.uuid4())
        return self.trace_id
    
    def _log(self, level: int, message: str, error_code: Optional[int] = None,
             context: Optional[Dict[str, Any]] = None, exc_info: bool = False) -> None:
        """
        Internal logging method with structured data.
        
        Args:
            level: Log level (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Log message
            error_code: Optional error code
            context: Optional context dictionary
            exc_info: Whether to include exception info
        """
        extra = {}
        
        if self.trace_id:
            extra['trace_id'] = self.trace_id
        
        if error_code:
            extra['error_code'] = error_code
        
        if context:
            extra['context'] = context
        
        self.logger.log(level, message, extra=extra, exc_info=exc_info)
    
    def debug(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, context=context)
    
    def info(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log info message."""
        self._log(logging.INFO, message, context=context)
    
    def warning(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, context=context)
    
    def error(self, message: str, error_code: Optional[int] = None,
              context: Optional[Dict[str, Any]] = None, exc_info: bool = False) -> None:
        """Log error message."""
        self._log(logging.ERROR, message, error_code=error_code, context=context, exc_info=exc_info)
    
    def critical(self, message: str, error_code: Optional[int] = None,
                 context: Optional[Dict[str, Any]] = None, exc_info: bool = False) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, message, error_code=error_code, context=context, exc_info=exc_info)
    
    def exception(self, message: str, error_code: Optional[int] = None,
                  context: Optional[Dict[str, Any]] = None) -> None:
        """Log exception with traceback."""
        self._log(logging.ERROR, message, error_code=error_code, context=context, exc_info=True)


# Global logger registry
_loggers: Dict[str, StructuredLogger] = {}


def setup_logging(level: str = "INFO", json_format: bool = True) -> None:
    """
    Set up logging configuration for the entire application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Whether to use JSON formatting (True) or plain text (False)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger("ml_scanner")
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    
    # Set formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    
    # Prevent propagation to root logger
    root_logger.propagate = False


def get_logger(name: str) -> StructuredLogger:
    """
    Get or create a structured logger for the given name.
    
    Args:
        name: Logger name (typically __name__ of the module)
        
    Returns:
        StructuredLogger instance
    """
    if name not in _loggers:
        # Ensure name starts with ml_scanner
        if not name.startswith("ml_scanner"):
            name = f"ml_scanner.{name}"
        
        logger = logging.getLogger(name)
        _loggers[name] = StructuredLogger(logger)
    
    return _loggers[name]


# Initialize logging on module import
setup_logging()

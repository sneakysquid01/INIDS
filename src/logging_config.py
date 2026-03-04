from __future__ import annotations

import logging

from flask import has_request_context, request


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if has_request_context():
            record.request_id = request.headers.get("X-Request-ID", "-")
            record.endpoint = request.path
        else:
            record.request_id = "-"
            record.endpoint = "-"
        return True


def configure_logging(level: int = logging.INFO) -> None:
    root_logger = logging.getLogger()
    if getattr(root_logger, "_inids_configured", False):
        return

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] request_id=%(request_id)s endpoint=%(endpoint)s %(message)s"
        )
    )
    handler.addFilter(RequestContextFilter())

    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
    root_logger._inids_configured = True


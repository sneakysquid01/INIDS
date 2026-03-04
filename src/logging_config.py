from __future__ import annotations

import logging

from flask import has_request_context, request


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if has_request_context():
            record.request_id = request.headers.get("X-Request-ID", "-")
            record.endpoint = request.path
            record.source_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "-")
        else:
            record.request_id = "-"
            record.endpoint = "-"
            record.source_ip = "-"
        record.risk_score = getattr(record, "risk_score", "-")
        record.action = getattr(record, "action", "-")
        return True


def configure_logging(level: int = logging.INFO) -> None:
    root_logger = logging.getLogger()
    if getattr(root_logger, "_inids_configured", False):
        return

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            (
                "timestamp=%(asctime)s level=%(levelname)s request_id=%(request_id)s "
                "source_ip=%(source_ip)s risk_score=%(risk_score)s action=%(action)s "
                "endpoint=%(endpoint)s message=%(message)s"
            )
        )
    )
    handler.addFilter(RequestContextFilter())

    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
    root_logger._inids_configured = True

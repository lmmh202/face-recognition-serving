"""JSON logs, one object per line on stdout."""

import json
import logging
import sys

logger = logging.getLogger("face_recognition_serving")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields = getattr(record, "fields", {})
        return json.dumps({"level": record.levelname, "event": record.getMessage(), **fields})


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False

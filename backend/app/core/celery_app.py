from __future__ import annotations

from celery import Celery

from app.core.config import settings


def make_celery() -> Celery:
    broker = settings.celery_broker_url or settings.redis_url
    backend = settings.celery_result_backend or settings.redis_url
    return Celery(
        "gap_architect",
        broker=broker,
        backend=backend,
        include=["app.tasks.gap_analysis"],
    )


celery_app = make_celery()
celery_app.conf.result_backend_transport_options = {
    "global_keyprefix": "gap_architect:",
}
try:
    celery_app.backend.task_keyprefix = b"gap_architect:task:"
    celery_app.backend.group_keyprefix = b"gap_architect:group:"
    celery_app.backend.chord_keyprefix = b"gap_architect:chord:"
except Exception:
    pass

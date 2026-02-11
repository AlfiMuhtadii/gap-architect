from __future__ import annotations

from uuid import UUID

from celery import shared_task
import asyncio

from app.services import llm_service


@shared_task(bind=True, max_retries=3, default_retry_delay=30)
def process_gap_analysis_task(self, gap_analysis_id: str) -> None:
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(llm_service.process_gap_analysis(UUID(gap_analysis_id)))
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
    except Exception as exc:  # noqa: BLE001
        raise self.retry(exc=exc)

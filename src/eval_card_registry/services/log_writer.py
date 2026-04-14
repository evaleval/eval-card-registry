"""
Async resolve log writer — buffers log entries in memory and periodically
flushes them to an HF Storage Bucket as partitioned parquet files.

Separate from the pipeline's resolution_log table. This tracks individual
API resolve requests for resolver improvement.
"""
from __future__ import annotations

import asyncio
import io
import logging
import threading
from datetime import datetime, timezone

import pandas as pd

from eval_card_registry.config import settings

logger = logging.getLogger(__name__)

_LOG_SCHEMA = {
    "request_id": "string",
    "raw_value": "string",
    "entity_type": "string",
    "source_config": "string",
    "canonical_id": "string",
    "strategy": "string",
    "confidence": "float64",
    "timestamp": "string",
}

# Drop oldest entries when buffer exceeds this size (prevents OOM on
# persistent flush failures). At ~200 bytes/entry this is ~2 MB.
_MAX_BUFFER_SIZE = 10_000


class ResolveLogWriter:
    """Thread-safe in-memory buffer that flushes to an HF Storage Bucket."""

    def __init__(self, bucket_id: str) -> None:
        self._bucket_id = bucket_id
        self._buffer: list[dict] = []
        self._lock = threading.Lock()
        self._flush_task: asyncio.Task | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._bucket_id)

    def append(self, entry: dict) -> None:
        """Append a log entry to the buffer. Thread-safe."""
        if not self.enabled:
            return
        with self._lock:
            self._buffer.append(entry)
            if len(self._buffer) > _MAX_BUFFER_SIZE:
                dropped = len(self._buffer) - _MAX_BUFFER_SIZE
                self._buffer = self._buffer[dropped:]
                logger.warning("Resolve log buffer full, dropped %d oldest entries", dropped)

    def start(self, interval_seconds: int) -> None:
        """Start the periodic flush background task."""
        if not self.enabled:
            return
        self._flush_task = asyncio.create_task(self._periodic_flush(interval_seconds))

    async def stop(self) -> None:
        """Stop the background task and do a final flush."""
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        # Final flush
        await self._flush()

    async def _periodic_flush(self, interval: int) -> None:
        while True:
            await asyncio.sleep(interval)
            await self._flush()

    async def _flush(self) -> None:
        """Flush buffered entries to the Storage Bucket as a parquet part file."""
        with self._lock:
            if not self._buffer:
                return
            entries = self._buffer.copy()
            self._buffer.clear()

        try:
            df = pd.DataFrame(entries)
            for col, dtype in _LOG_SCHEMA.items():
                if col in df.columns:
                    df[col] = df[col].astype(dtype)

            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            parquet_bytes = buf.getvalue()

            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            filename = f"api_resolve_log/part-{ts}.parquet"

            from huggingface_hub import HfApi
            api = HfApi(token=settings.hf_token or None)
            await asyncio.to_thread(
                api.batch_bucket_files,
                self._bucket_id,
                add=[(parquet_bytes, filename)],
            )
            logger.info("Flushed %d resolve log entries to %s/%s", len(entries), self._bucket_id, filename)
        except Exception:
            logger.exception("Failed to flush resolve log entries")
            # Re-add failed entries to the front of the buffer. The buffer
            # cap in append() prevents unbounded growth if flushes keep
            # failing — oldest entries are dropped when the cap is hit.
            with self._lock:
                self._buffer = entries + self._buffer
                if len(self._buffer) > _MAX_BUFFER_SIZE:
                    self._buffer = self._buffer[-_MAX_BUFFER_SIZE:]

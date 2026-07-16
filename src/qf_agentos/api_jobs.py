"""In-process async job queue for the REST API.

``POST /jobs`` submits a solve to a bounded thread pool and returns immediately
with a job id; ``GET /jobs/{id}`` polls status and returns the result when done.
This keeps the event loop responsive for long QAOA runs without an external
broker. It is single-instance / in-memory by design: for a multi-replica
deployment, back the same interface with a real queue (Celery / RQ / Arq).
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .core.observability import get_logger

_logger = get_logger("api.jobs")


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    status: JobStatus
    created_at: float
    problem: str = ""
    started_at: float | None = None
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    @property
    def done(self) -> bool:
        return self.status in (JobStatus.SUCCEEDED, JobStatus.FAILED)

    def summary(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "status": self.status.value,
            "problem": self.problem,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }


class JobManager:
    """A bounded thread pool + an LRU-capped registry of job records."""

    def __init__(self, *, workers: int, max_jobs: int) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, workers), thread_name_prefix="qf-job"
        )
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        self._max_jobs = max(1, max_jobs)

    def submit(self, fn: Callable[[], dict[str, Any]], *, problem: str = "") -> Job:
        job = Job(
            id=uuid.uuid4().hex,
            status=JobStatus.QUEUED,
            created_at=time.time(),
            problem=problem,
        )
        with self._lock:
            self._jobs[job.id] = job
            self._evict_locked()
        self._executor.submit(self._run, job, fn)
        return job

    def _run(self, job: Job, fn: Callable[[], dict[str, Any]]) -> None:
        with self._lock:
            job.started_at = time.time()
            job.status = JobStatus.RUNNING
        try:
            result = fn()
        except Exception as exc:
            with self._lock:
                job.error = str(exc)
                job.status = JobStatus.FAILED
            _logger.warning("job %s failed: %s", job.id, exc)
        else:
            with self._lock:
                job.result = result
                job.status = JobStatus.SUCCEEDED
        finally:
            with self._lock:
                job.finished_at = time.time()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def _evict_locked(self) -> None:
        """Drop the oldest COMPLETED jobs once the retention cap is exceeded."""
        while len(self._jobs) > self._max_jobs:
            for jid, j in self._jobs.items():
                if j.done:
                    del self._jobs[jid]
                    break
            else:
                break  # nothing evictable yet (all still running)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


__all__ = ["Job", "JobManager", "JobStatus"]

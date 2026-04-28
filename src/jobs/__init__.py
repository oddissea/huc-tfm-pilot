"""Cola de inferencia: gestiona uploads pendientes en disco efímero.

`manager.JobManager` maneja la persistencia (UUID + meta.json + raw file en
`/tmp/queue/<uuid>/`); `worker.start_worker` lanza un thread daemon que
consume la cola FIFO.
"""

from .manager import Job, JobManager, JobStatus
from .worker import start_worker

__all__ = ["Job", "JobManager", "JobStatus", "start_worker"]

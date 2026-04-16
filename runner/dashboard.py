"""Rich Live dashboard tracking (domain, model) progress in a single table."""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table


@dataclass
class _TaskState:
    total: int
    done: int = 0
    errors: int = 0
    brier_sum: float = 0.0
    brier_n: int = 0
    progress_task_id: int | None = None


class Dashboard:
    def __init__(self, title: str):
        self.console = Console()
        self.title = title
        self._tasks: dict[tuple[str, str], _TaskState] = {}
        self._progress = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
        )
        self._live: Live | None = None

    def register(self, domain: str, model: str, total: int) -> None:
        key = (domain, model)
        pid = self._progress.add_task(f"{domain} · {model}", total=total)
        self._tasks[key] = _TaskState(total=total, progress_task_id=pid)

    def record(self, domain: str, model: str, *, error: bool, brier: float | None) -> None:
        st = self._tasks[(domain, model)]
        st.done += 1
        if error:
            st.errors += 1
        if brier is not None and brier == brier:  # not NaN
            st.brier_sum += brier
            st.brier_n += 1
        self._progress.update(st.progress_task_id, completed=st.done)

    def _render(self) -> Group:
        table = Table(title=self.title, expand=False)
        table.add_column("Domain")
        table.add_column("Model")
        table.add_column("Done", justify="right")
        table.add_column("Err", justify="right")
        table.add_column("Brier", justify="right")
        for (domain, model), st in self._tasks.items():
            brier = f"{st.brier_sum / st.brier_n:.3f}" if st.brier_n else "—"
            table.add_row(domain, model, f"{st.done}/{st.total}", str(st.errors), brier)
        return Group(table, self._progress)

    def __enter__(self):
        self._live = Live(self._render(), console=self.console, refresh_per_second=8)
        self._live.__enter__()
        return self

    def refresh(self) -> None:
        if self._live:
            self._live.update(self._render())

    def __exit__(self, *exc):
        self.refresh()
        assert self._live is not None
        self._live.__exit__(*exc)

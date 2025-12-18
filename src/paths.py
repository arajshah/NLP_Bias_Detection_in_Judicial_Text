from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def cap_dir(self) -> Path:
        return self.root / "cap_data"

    @property
    def models_dir(self) -> Path:
        return self.root / "artifacts" / "models"

    @property
    def reports_dir(self) -> Path:
        return self.root / "artifacts" / "reports"


def find_project_root(start: Path | None = None) -> Path:
    """Find project root by walking upward until `data/` and `notebooks/` exist."""
    cur = (start or Path.cwd()).resolve()
    for _ in range(10):
        if (cur / "data").exists() and (cur / "notebooks").exists():
            return cur
        cur = cur.parent
    # Fallback to cwd if detection fails
    return (start or Path.cwd()).resolve()


def get_paths(start: Path | None = None) -> ProjectPaths:
    root = find_project_root(start)
    return ProjectPaths(root=root)



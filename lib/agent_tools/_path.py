import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
_philo_dir = _project_root / "philo"

for p in [str(_project_root), str(_philo_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

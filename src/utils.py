import subprocess
import sys
from pathlib import Path


def launch_tensorboard(logdir):
    """Launch tensorboard in background."""
    logdir = Path(logdir).absolute()
    logdir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.Popen(
            [sys.executable, "-m", "tensorboard.main", "--logdir", str(logdir)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:
        pass

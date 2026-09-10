"""Persistence layer: the task list is a JSON file in the home directory."""

import json
import os
import tempfile

_PATH = os.path.expanduser("~/.taskcli.json")


def store_path():
    """Return the on-disk location of the task file."""
    return _PATH


def load_tasks():
    """Read the task list from disk, returning [] when the file is absent."""
    if not os.path.exists(_PATH):
        return []
    with open(_PATH) as fh:
        return json.load(fh)


def save_tasks(tasks):
    """Persist the task list to disk atomically as indented JSON.

    The data is written to a temp file in the same directory and then
    renamed over the target, so a crash never leaves a half-written file.
    """
    directory = os.path.dirname(_PATH) or "."
    fd, tmp = tempfile.mkstemp(dir=directory)
    with os.fdopen(fd, "w") as fh:
        json.dump(tasks, fh, indent=2)
    os.replace(tmp, _PATH)


def clear_tasks():
    """Delete the task file entirely."""
    if os.path.exists(_PATH):
        os.remove(_PATH)


def backup_tasks(dest):
    """Copy the current task file to `dest`."""
    tasks = load_tasks()
    with open(dest, "w") as fh:
        json.dump(tasks, fh, indent=2)


def task_count():
    """Number of tasks currently stored."""
    return len(load_tasks())

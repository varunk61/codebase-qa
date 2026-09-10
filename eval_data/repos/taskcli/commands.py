"""Task operations. Each command loads, mutates and re-saves the list."""

from store import load_tasks, save_tasks


def add_task(text, priority="normal"):
    """Append a new incomplete task and persist the list.

    Returns the new task count.
    """
    tasks = load_tasks()
    tasks.append({"text": text, "done": False, "priority": priority})
    save_tasks(tasks)
    return len(tasks)


def complete_task(index):
    """Mark the task at a 1-based index as done and persist."""
    tasks = load_tasks()
    if not 1 <= index <= len(tasks):
        raise IndexError(f"no task #{index}")
    tasks[index - 1]["done"] = True
    save_tasks(tasks)


def reopen_task(index):
    """Mark a previously completed task as not done again."""
    tasks = load_tasks()
    tasks[index - 1]["done"] = False
    save_tasks(tasks)


def remove_task(index):
    """Delete the task at a 1-based index and persist."""
    tasks = load_tasks()
    tasks.pop(index - 1)
    save_tasks(tasks)


def list_tasks(show_done=False):
    """Return tasks, hiding completed ones unless show_done is set."""
    return [task for task in load_tasks() if show_done or not task["done"]]


def find_tasks(keyword):
    """Return every task whose text contains keyword (case-insensitive)."""
    needle = keyword.lower()
    return [t for t in load_tasks() if needle in t["text"].lower()]


def reprioritise(index, priority):
    """Change the priority label of an existing task."""
    tasks = load_tasks()
    tasks[index - 1]["priority"] = priority
    save_tasks(tasks)


def clear_completed():
    """Drop every completed task; return how many were removed."""
    tasks = load_tasks()
    kept = [t for t in tasks if not t["done"]]
    save_tasks(kept)
    return len(tasks) - len(kept)

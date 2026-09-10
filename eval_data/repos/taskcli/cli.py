"""Command-line front end: argument parsing and subcommand dispatch."""

import sys

from commands import (
    add_task,
    clear_completed,
    complete_task,
    find_tasks,
    list_tasks,
    remove_task,
)

USAGE = "usage: task [add TEXT | done N | rm N | list [--all] | find TERM | clean]"


def parse_args(argv):
    """Split argv into a (command, remaining-args) pair; default to 'list'.

    No third-party parser is used - the grammar is small enough to handle
    positionally.
    """
    if not argv:
        return "list", []
    return argv[0], argv[1:]


def _print_tasks(tasks):
    """Render a task list to stdout with a done/undone marker."""
    for i, task in enumerate(tasks, start=1):
        mark = "x" if task["done"] else " "
        print(f"{i:>2}. [{mark}] {task['text']}")


def dispatch(command, rest):
    """Route a parsed command to its handler and return an exit code."""
    if command == "add":
        add_task(" ".join(rest))
    elif command == "done":
        complete_task(int(rest[0]))
    elif command == "rm":
        remove_task(int(rest[0]))
    elif command == "list":
        _print_tasks(list_tasks(show_done="--all" in rest))
    elif command == "find":
        _print_tasks(find_tasks(" ".join(rest)))
    elif command == "clean":
        removed = clear_completed()
        print(f"removed {removed} completed task(s)")
    else:
        print(USAGE, file=sys.stderr)
        return 2
    return 0


def main(argv=None):
    """Entry point: parse args, dispatch the subcommand, and set exit status."""
    command, rest = parse_args(sys.argv[1:] if argv is None else argv)
    raise SystemExit(dispatch(command, rest))

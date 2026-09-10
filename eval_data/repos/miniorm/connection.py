"""Connection wrapper and a Query subclass that can execute itself."""

import sqlite3

from query import Query


class Connection:
    """Thin wrapper over a sqlite3 connection with a Query factory."""

    def __init__(self, path=":memory:"):
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")

    def query(self, table):
        """Start a new BoundQuery against this connection's database."""
        return BoundQuery(table, self)

    def execute(self, sql, params=()):
        """Run raw SQL and return all rows as a list."""
        cur = self._conn.execute(sql, params)
        return cur.fetchall()

    def execute_one(self, sql, params=()):
        """Run raw SQL and return a single row, or None."""
        cur = self._conn.execute(sql, params)
        return cur.fetchone()

    def commit(self):
        """Flush the current transaction."""
        self._conn.commit()

    def close(self):
        """Close the underlying connection."""
        self._conn.close()


class BoundQuery(Query):
    """A Query that knows how to run itself against a Connection."""

    def __init__(self, table, connection):
        super().__init__(table)
        self._connection = connection

    def all(self):
        """Build the SQL and return every matching row."""
        sql, params = self.build()
        return self._connection.execute(sql, params)

    def first(self):
        """Return only the first matching row, or None."""
        rows = self.limit(1).all()
        return rows[0] if rows else None

    def count(self):
        """Return the number of rows the current predicates match."""
        sql, params = self.count_query()
        row = self._connection.execute_one(sql, params)
        return row[0] if row else 0

    def exists(self):
        """True when at least one row matches the current predicates."""
        return self.count() > 0

    def pluck(self, column):
        """Return a flat list of one column's values across all matches."""
        return [row[column] for row in self.select(column).all()]

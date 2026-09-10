"""A fluent, parameterised SQL SELECT builder."""


class Query:
    """Collects clauses and compiles them to a parameterised SQL SELECT."""

    def __init__(self, table):
        self.table = table
        self._columns = ["*"]
        self._where = []
        self._params = []
        self._order = None
        self._limit = None
        self._offset = None

    def select(self, *columns):
        """Restrict the projection to the named columns."""
        self._columns = list(columns) or ["*"]
        return self

    def where(self, column, value):
        """Add an equality predicate; value is bound as a parameter, not inlined.

        This is the single place user input enters the SQL, and it never does
        so by string interpolation - only as a bound `?` placeholder.
        """
        self._where.append(f"{column} = ?")
        self._params.append(value)
        return self

    def where_in(self, column, values):
        """Add an `IN (...)` predicate with one bound parameter per value."""
        marks = ", ".join("?" for _ in values)
        self._where.append(f"{column} IN ({marks})")
        self._params.extend(values)
        return self

    def order_by(self, column, desc=False):
        """Set the ORDER BY clause."""
        self._order = f"{column} {'DESC' if desc else 'ASC'}"
        return self

    def limit(self, n):
        """Cap the number of returned rows."""
        self._limit = int(n)
        return self

    def offset(self, n):
        """Skip the first n rows of the result."""
        self._offset = int(n)
        return self

    def build(self):
        """Compile the collected clauses into an (sql, params) tuple."""
        sql = f"SELECT {', '.join(self._columns)} FROM {self.table}"
        if self._where:
            sql += " WHERE " + " AND ".join(self._where)
        if self._order:
            sql += f" ORDER BY {self._order}"
        if self._limit is not None:
            sql += f" LIMIT {self._limit}"
        if self._offset is not None:
            sql += f" OFFSET {self._offset}"
        return sql, tuple(self._params)

    def count_query(self):
        """Return an (sql, params) tuple that counts matching rows."""
        sql = f"SELECT COUNT(*) FROM {self.table}"
        if self._where:
            sql += " WHERE " + " AND ".join(self._where)
        return sql, tuple(self._params)

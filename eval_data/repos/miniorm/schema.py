"""DDL and row-writing helpers built on top of Connection."""


def create_table(conn, name, columns):
    """Issue CREATE TABLE IF NOT EXISTS from a {column: type} mapping."""
    cols = ", ".join(f"{col} {typ}" for col, typ in columns.items())
    conn.execute(f"CREATE TABLE IF NOT EXISTS {name} ({cols})")


def drop_table(conn, name):
    """Drop a table if it exists."""
    conn.execute(f"DROP TABLE IF EXISTS {name}")


def add_column(conn, table, column, typ):
    """Append a column to an existing table."""
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {typ}")


def table_exists(conn, name):
    """True when a table of the given name is present in the schema."""
    row = conn.execute_one(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    )
    return row is not None


def insert_row(conn, table, values):
    """Insert one row from a {column: value} dict, binding values as params."""
    keys = ", ".join(values)
    marks = ", ".join("?" for _ in values)
    conn.execute(
        f"INSERT INTO {table} ({keys}) VALUES ({marks})",
        tuple(values.values()),
    )


def insert_many(conn, table, rows):
    """Insert a list of {column: value} dicts that share the same keys."""
    if not rows:
        return
    keys = list(rows[0])
    marks = ", ".join("?" for _ in keys)
    for row in rows:
        conn.execute(
            f"INSERT INTO {table} ({', '.join(keys)}) VALUES ({marks})",
            tuple(row[k] for k in keys),
        )


def upsert_row(conn, table, key_column, values):
    """Insert a row, or replace the existing one with the same key value."""
    keys = ", ".join(values)
    marks = ", ".join("?" for _ in values)
    conn.execute(
        f"INSERT OR REPLACE INTO {table} ({keys}) VALUES ({marks})",
        tuple(values.values()),
    )

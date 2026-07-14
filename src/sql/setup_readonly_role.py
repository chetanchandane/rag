"""
Create/refresh the read-only Postgres role for the query path (governance).

Runs the equivalent of roles.sql via psycopg (no psql required), connecting as
the OWNER (settings.database_url). Idempotent: creates the role if missing,
otherwise resets its password. Prints the DATABASE_URL_RO to paste into .env.

Usage
-----
    python -m src.sql.setup_readonly_role                 # generate a password
    python -m src.sql.setup_readonly_role --password ...  # supply your own
"""

from __future__ import annotations

import argparse
import secrets
import string
import sys
from urllib.parse import urlparse, urlunparse

from src.sql.settings import settings

ROLE = "clinrag_readonly"


def _gen_password(n: int = 24) -> str:
    # URL-safe alphabet only (no chars that need percent-encoding in a URL).
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(n))


def _readonly_url(owner_url: str, role: str, password: str) -> str:
    u = urlparse(owner_url)
    port = f":{u.port}" if u.port else ""
    netloc = f"{role}:{password}@{u.hostname}{port}"
    return urlunparse((u.scheme, netloc, u.path, u.params, u.query, u.fragment))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the read-only DB role.")
    parser.add_argument("--password", default=None, help="role password (generated if omitted)")
    args = parser.parse_args()

    if not settings.database_url:
        sys.exit("DATABASE_URL (owner) is not set in .env")

    password = args.password or _gen_password()

    import psycopg
    from psycopg import sql

    with psycopg.connect(settings.database_url, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("SELECT current_database()")
        dbname = cur.fetchone()[0]

        cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (ROLE,))
        role_ident = sql.Identifier(ROLE)
        if cur.fetchone():
            cur.execute(sql.SQL("ALTER ROLE {} LOGIN PASSWORD {}").format(role_ident, sql.Literal(password)))
            action = "updated"
        else:
            cur.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(role_ident, sql.Literal(password)))
            action = "created"

        cur.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(sql.Identifier(dbname), role_ident))
        cur.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(role_ident))
        cur.execute(sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA public TO {}").format(role_ident))
        cur.execute(
            sql.SQL("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO {}").format(role_ident)
        )
        cur.execute(
            sql.SQL("REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON ALL TABLES IN SCHEMA public FROM {}").format(role_ident)
        )
        cur.execute(
            sql.SQL("ALTER ROLE {} SET statement_timeout = {}").format(
                role_ident, sql.Literal(settings.sql_statement_timeout_ms)
            )
        )
        cur.execute(sql.SQL("ALTER ROLE {} SET default_transaction_read_only = on").format(role_ident))

    ro_url = _readonly_url(settings.database_url, ROLE, password)
    print(f"Read-only role '{ROLE}' {action} on database '{dbname}'.")
    print(f"  statement_timeout = {settings.sql_statement_timeout_ms}ms, default_transaction_read_only = on")
    print("\nAdd this line to your .env:\n")
    print(f"DATABASE_URL_RO={ro_url}")


if __name__ == "__main__":
    main()

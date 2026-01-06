"""
Database Adapter - Cross-database compatibility layer

Provides a unified interface for both SQLite and PostgreSQL connections.
SQLite uses conn.execute() with ? placeholders.
PostgreSQL uses cursor.execute() with %s placeholders.

This adapter normalizes the interface so calling code can work with both.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Iterator, Optional, Union
from contextlib import contextmanager


class DBAdapter:
    """
    Unified database adapter for SQLite and PostgreSQL.
    
    Usage:
        adapter = DBAdapter(conn)
        rows = adapter.execute("SELECT * FROM foo WHERE id = ?", (123,)).fetchall()
        adapter.executemany("INSERT INTO bar VALUES (?, ?)", data)
        adapter.commit()
    """
    
    def __init__(self, conn: Any, db_type: Optional[str] = None):
        """
        Initialize adapter.
        
        Args:
            conn: Database connection (sqlite3.Connection or psycopg2.connection)
            db_type: 'sqlite' or 'postgresql' (auto-detected if not provided)
        """
        self.conn = conn
        self.db_type = db_type or self._detect_db_type(conn)
        self._cursor = None
    
    def _detect_db_type(self, conn: Any) -> str:
        """Auto-detect database type from connection."""
        conn_type = type(conn).__module__
        if 'sqlite' in conn_type:
            return 'sqlite'
        elif 'psycopg2' in conn_type:
            return 'postgresql'
        else:
            # Check for SQLAlchemy connections
            if hasattr(conn, 'dialect'):
                dialect_name = conn.dialect.name
                if 'sqlite' in dialect_name:
                    return 'sqlite'
                elif 'postgres' in dialect_name:
                    return 'postgresql'
            # Default to SQLite syntax
            return 'sqlite'
    
    def _convert_placeholders(self, sql: str) -> str:
        """Convert ? placeholders to %s for PostgreSQL."""
        if self.db_type == 'postgresql':
            return sql.replace('?', '%s')
        return sql
    
    @property
    def cursor(self):
        """Get or create cursor."""
        if self._cursor is None:
            self._cursor = self.conn.cursor()
        return self._cursor
    
    def execute(self, sql: str, params: tuple = ()) -> 'DBAdapter':
        """
        Execute a SQL statement.
        
        Args:
            sql: SQL statement with ? placeholders
            params: Parameters tuple
        
        Returns:
            self (for chaining)
        """
        converted_sql = self._convert_placeholders(sql)
        
        if self.db_type == 'sqlite':
            # SQLite can use conn.execute() directly
            self._result = self.conn.execute(converted_sql, params)
        else:
            # PostgreSQL needs cursor.execute()
            self.cursor.execute(converted_sql, params)
            self._result = self.cursor
        
        return self
    
    def executemany(self, sql: str, params_list: list) -> 'DBAdapter':
        """
        Execute a SQL statement with multiple parameter sets.
        
        Args:
            sql: SQL statement with ? placeholders
            params_list: List of parameter tuples
        
        Returns:
            self (for chaining)
        """
        converted_sql = self._convert_placeholders(sql)
        
        if self.db_type == 'sqlite':
            self.conn.executemany(converted_sql, params_list)
        else:
            self.cursor.executemany(converted_sql, params_list)
        
        return self
    
    def fetchall(self) -> list:
        """Fetch all results from last execute."""
        if hasattr(self, '_result'):
            return self._result.fetchall()
        return []
    
    def fetchone(self) -> Optional[tuple]:
        """Fetch one result from last execute."""
        if hasattr(self, '_result'):
            return self._result.fetchone()
        return None
    
    def commit(self):
        """Commit the transaction."""
        self.conn.commit()
    
    def rollback(self):
        """Rollback the transaction."""
        self.conn.rollback()
    
    def close(self):
        """Close cursor and connection."""
        if self._cursor:
            self._cursor.close()
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.commit()
        self.close()


def wrap_connection(conn: Any, db_type: Optional[str] = None) -> DBAdapter:
    """
    Wrap a database connection in a DBAdapter.
    
    Args:
        conn: Raw database connection
        db_type: Optional hint for database type
    
    Returns:
        DBAdapter wrapping the connection
    """
    return DBAdapter(conn, db_type)

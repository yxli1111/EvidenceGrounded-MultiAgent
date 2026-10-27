import hashlib
import json
import sqlite3
import time
from pathlib import Path


class SQLiteCache:
    """Persistent SQLite cache for successful LLM responses."""

    def __init__(
        self,
        db_path: str = "cache/llm_cache.db",
        ttl_seconds: int = 86400,
    ):
        if ttl_seconds <= 0:
            raise ValueError("TTL must be positive")

        self.db_path = Path(db_path)
        self.ttl_seconds = ttl_seconds

        # Step 1: Create cache directory.
        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Step 2: Initialize SQLite table.
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    response_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)

    @staticmethod
    def make_key(
        *,
        stage: str,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Create a stable hash from request parameters."""

        payload = {
            "stage": stage,
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Serialize parameters consistently.
        serialized = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        # Hash the request without storing the prompt as a key.
        return hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

    def get(self, cache_key: str) -> dict | None:
        """Return cached response, or None on a cache miss."""

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT response_json, created_at
                FROM llm_cache
                WHERE cache_key = ?
                """,
                (cache_key,),
            ).fetchone()

            if row is None:
                return None

            response_json, created_at = row

            # Step 3: Check TTL expiration.
            if time.time() - created_at >= self.ttl_seconds:
                conn.execute(
                    "DELETE FROM llm_cache WHERE cache_key = ?",
                    (cache_key,),
                )
                return None

            return json.loads(response_json)

    def set(self, cache_key: str, response: dict) -> None:
        """Store a successful response."""

        serialized = json.dumps(
            response,
            ensure_ascii=False,
            allow_nan=False,
        )

        # Step 4: Insert or update cache entry.
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO llm_cache
                (cache_key, response_json, created_at)
                VALUES (?, ?, ?)
                """,
                (cache_key, serialized, time.time()),
            )

    def delete(self, cache_key: str) -> None:
        """Remove a cached response."""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM llm_cache WHERE cache_key = ?",
                (cache_key,),
            )

    def clear(self) -> None:
        """Remove all cached responses."""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM llm_cache")

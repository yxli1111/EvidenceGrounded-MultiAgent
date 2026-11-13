import sqlite3
import time

import pytest

from src.core.cache import SQLiteCache


def test_cache_miss(tmp_path):
    cache = SQLiteCache(
        db_path=str(tmp_path / "cache.db")
    )

    assert cache.get("missing-key") is None


def test_cache_set_and_get(tmp_path):
    cache = SQLiteCache(
        db_path=str(tmp_path / "cache.db")
    )

    response = {
        "text": "Hello DSA4213",
        "model": "openai/gpt-4o-mini",
    }

    cache.set("test-key", response)

    assert cache.get("test-key") == response


def test_cache_persistence(tmp_path):
    db_path = str(tmp_path / "cache.db")

    first_cache = SQLiteCache(db_path=db_path)
    first_cache.set("key", {"text": "Persistent"})

    # Reopen the database with another cache instance.
    second_cache = SQLiteCache(db_path=db_path)

    assert second_cache.get("key") == {
        "text": "Persistent"
    }


def test_cache_key_consistency():
    params = {
        "stage": "claim_extraction",
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "temperature": 0,
        "max_tokens": 128,
    }

    key1 = SQLiteCache.make_key(**params)
    key2 = SQLiteCache.make_key(**params)

    assert key1 == key2
    assert len(key1) == 64


def test_cache_key_changes_with_model():
    params = {
        "stage": "claim_extraction",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "temperature": 0,
        "max_tokens": 128,
    }

    key1 = SQLiteCache.make_key(
        model="openai/gpt-4o-mini",
        **params,
    )

    key2 = SQLiteCache.make_key(
        model="google/gemini-2.0-flash",
        **params,
    )

    assert key1 != key2


def test_cache_expiration(tmp_path):
    db_path = str(tmp_path / "cache.db")

    cache = SQLiteCache(
        db_path=db_path,
        ttl_seconds=10,
    )

    cache.set("expired-key", {"text": "Old"})

    # Simulate an expired entry without waiting.
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE llm_cache
            SET created_at = ?
            WHERE cache_key = ?
            """,
            (time.time() - 20, "expired-key"),
        )

    assert cache.get("expired-key") is None


def test_cache_delete(tmp_path):
    cache = SQLiteCache(
        db_path=str(tmp_path / "cache.db")
    )

    cache.set("key", {"text": "Hello"})
    cache.delete("key")

    assert cache.get("key") is None


def test_cache_clear(tmp_path):
    cache = SQLiteCache(
        db_path=str(tmp_path / "cache.db")
    )

    cache.set("key1", {"text": "A"})
    cache.set("key2", {"text": "B"})

    cache.clear()

    assert cache.get("key1") is None
    assert cache.get("key2") is None


def test_invalid_ttl(tmp_path):
    with pytest.raises(ValueError):
        SQLiteCache(
            db_path=str(tmp_path / "cache.db"),
            ttl_seconds=0,
        )

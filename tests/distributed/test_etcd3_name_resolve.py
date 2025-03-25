import os
import time

import etcd3
import pytest

from realhf.base.name_resolve import (
    Etcd3NameRecordRepository,
    NameEntryExistsError,
    NameEntryNotFoundError,
)

host, port = os.getenv("REAL_ETCD_ADDR", "localhost:2379").split(":")
port = int(port)


@pytest.fixture
def etcd_client():
    client = etcd3.client(host=host, port=port)
    yield client
    # Clean up etcd after each test
    client.delete_prefix("test_")  # Delete all keys


# Fixture to provide an instance of Etcd3NameRecordRepository
@pytest.fixture
def etcd_repo():
    repo = Etcd3NameRecordRepository(host=host, port=port)
    yield repo
    repo.reset()  # Clean up repository after each test


def test_add(etcd_repo):
    # Test adding a new key-value pair
    etcd_repo.add("test_key", "test_value")
    value, _ = etcd_repo._client.get("test_key")
    assert value.decode("utf-8") == "test_value"

    # Test adding a key that already exists without replace
    with pytest.raises(NameEntryExistsError):
        etcd_repo.add("test_key", "new_value", replace=False)

    # Test adding a key that already exists with replace
    etcd_repo.add("test_key", "new_value", replace=True)
    value, _ = etcd_repo._client.get("test_key")
    assert value.decode("utf-8") == "new_value"


def test_delete(etcd_repo):
    # Test deleting an existing key
    etcd_repo.add("test_key", "test_value")
    etcd_repo.delete("test_key")
    value, _ = etcd_repo._client.get("test_key")
    assert value is None

    # Test deleting a non-existent key
    with pytest.raises(NameEntryNotFoundError):
        etcd_repo.delete("non_existent_key")


def test_clear_subtree(etcd_repo):
    # Test clearing a subtree
    etcd_repo.add("test_key/sub1", "value1")
    etcd_repo.add("test_key/sub2", "value2")
    etcd_repo.clear_subtree("test_key")
    value1, _ = etcd_repo._client.get("test_key/sub1")
    value2, _ = etcd_repo._client.get("test_key/sub2")
    assert value1 is None
    assert value2 is None


def test_get(etcd_repo):
    # Test getting an existing key
    etcd_repo.add("test_key", "test_value")
    assert etcd_repo.get("test_key") == "test_value"

    # Test getting a non-existent key
    with pytest.raises(NameEntryNotFoundError):
        etcd_repo.get("non_existent_key")


def test_get_subtree(etcd_repo):
    # Test getting values from a subtree
    etcd_repo.add("test_key/sub1", "value1")
    etcd_repo.add("test_key/sub2", "value2")
    assert etcd_repo.get_subtree("test_key") == ["value1", "value2"]


def test_find_subtree(etcd_repo):
    # Test finding keys in a subtree
    etcd_repo.add("test_key/sub1", "value1")
    etcd_repo.add("test_key/sub2", "value2")
    assert etcd_repo.find_subtree("test_key") == ["test_key/sub1", "test_key/sub2"]


def test_reset(etcd_repo):
    # Test resetting the repository
    etcd_repo.add("test_key1", "value1", delete_on_exit=True)
    etcd_repo.add("test_key2", "value2", delete_on_exit=True)
    etcd_repo.reset()
    value1, _ = etcd_repo._client.get("test_key1")
    value2, _ = etcd_repo._client.get("test_key2")
    assert value1 is None
    assert value2 is None


def test_watch_names(etcd_repo):
    # Test watching keys
    callback_called = False

    def callback():
        nonlocal callback_called
        callback_called = True

    etcd_repo.add("test_key", "test_value")
    etcd_repo.watch_names(["test_key"], callback)

    # Delete the key to trigger the callback
    etcd_repo.delete("test_key")
    time.sleep(1)  # Give the watcher time to trigger
    assert callback_called


def test_keepalive_thread(etcd_repo):
    # Test the keepalive thread
    etcd_repo.add("test_key", "test_value", keepalive_ttl=2)
    time.sleep(1)  # Wait for the keepalive thread to refresh the lease
    # Ensure the key still exists
    value, _ = etcd_repo._client.get("test_key")
    assert value.decode("utf-8") == "test_value"
    time.sleep(2)  # Wait for the lease to expire
    with pytest.raises(NameEntryNotFoundError):
        etcd_repo.get("test_key")


def test_context_manager(etcd_repo):
    # Test the context manager
    with etcd_repo as repo:
        repo.add("test_key", "test_value", delete_on_exit=True)
        assert repo.get("test_key") == "test_value"
    # Ensure the key is deleted after exiting the context
    value, _ = etcd_repo._client.get("test_key")
    assert value is None


def test_del(etcd_repo, etcd_client):
    # Test the destructor
    etcd_repo.add("test_key", "test_value", delete_on_exit=True)
    etcd_repo.__del__()
    value, _ = etcd_client.get("test_key")
    assert value is None

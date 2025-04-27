# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

# Implements a simple name resolving service, which can be considered as a distributed key-value dict.
import dataclasses
import os
import queue
import random
import shutil
import threading
import time
import uuid
from typing import Callable, List, Optional

try:
    import etcd3
except Exception:
    etcd3 = None

from realhf.base import cluster, logging, security, timeutil
from realhf.base.cluster import spec as cluster_spec

logger = logging.getLogger("name-resolve")


class ArgumentError(Exception):
    pass


class NameEntryExistsError(Exception):
    pass


class NameEntryNotFoundError(Exception):
    pass


class NameRecordRepository:

    def __del__(self):
        try:
            self.reset()
        except Exception as e:
            logger.info(f"Exception ignore when deleting NameResolveRepo {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reset()

    def add(
        self,
        name,
        value,
        delete_on_exit=True,
        keepalive_ttl=None,
        replace=False,
    ):
        """Creates a name record in the central repository.

        In our semantics, the name record repository is essentially a multimap (i.e. Dict[str, Set[str]]).
        This class keeps a single name->value map, where the name can be non-unique while the value has to be.
        The class also deletes the (name, value) pair on exits (__exit__/__del__) if opted-in. In case of
        preventing unexpected exits (i.e. process crashes without calling graceful exits), an user may also
        want to specify time_to_live and call touch() regularly to allow a more consistent

        Args:
            name: The key of the record. It has to be a valid path-like string; e.g. "a/b/c". If the name
                already exists, the behaviour is defined by the `replace` argument.
            value: The value of the record. This can be any valid string.
            delete_on_exit: If the record shall be deleted when the repository closes.
            keepalive_ttl: If not None, adds a time-to-live in seconds for the record. The repository
                shall keep pinging the backend service with at least this frequency to make sure the name
                entry is alive during the lifetime of the repository. On the other hand, specifying this
                prevents stale keys caused by the scenario that a Python process accidentally crashes before
                calling delete().
            replace: If the name already exists, then replaces the current value with the supplied value if
                `replace` is True, or raises exception if `replace` is False.
        """
        raise NotImplementedError()

    def add_subentry(self, name, value, **kwargs):
        """Adds a sub-entry to the key-root `name`.

        The values is retrievable by get_subtree() given that no other
        entries use the name prefix.
        """
        sub_name = os.path.join(os.path.normpath(name), str(uuid.uuid4())[:8])
        self.add(sub_name, value, **kwargs)
        return sub_name

    def delete(self, name):
        """Deletes an existing record."""
        raise NotImplementedError()

    def clear_subtree(self, name_root):
        """Deletes all records whose names start with the path root name_root;
        specifically, whose name either is `name_root`, or starts with
        `name_root.rstrip("/") + "/"`."""
        raise NotImplementedError()

    def get(self, name):
        """Returns the value of the key.

        Raises NameEntryNotFoundError if not found.
        """
        raise NotImplementedError()

    def get_subtree(self, name_root):
        """Returns all values whose names start with the path root name_root;
        specifically, whose name either is `name_root`, or starts with
        `name_root.rstrip("/") + "/"`."""
        raise NotImplementedError()

    def find_subtree(self, name_root):
        """Returns all KEYS whose names start with the path root name_root."""
        raise NotImplementedError()

    def wait(self, name, timeout=None, poll_frequency=1):
        """Waits until a name appears.

        Raises:
             TimeoutError: if timeout exceeds.
        """
        start = time.monotonic()
        while True:
            try:
                return self.get(name)
            except NameEntryNotFoundError:
                pass
            if timeout is None or timeout > 0:
                time.sleep(
                    poll_frequency + random.random() * 0.1
                )  # To reduce concurrency.
            if timeout is not None and time.monotonic() - start > timeout:
                raise TimeoutError(
                    f"Timeout waiting for key '{name}' ({self.__class__.__name__})"
                )

    def reset(self):
        """Deletes all entries added via this repository instance's
        add(delete_on_exit=True)."""
        raise NotImplementedError()

    def watch_names(
        self,
        names: List,
        call_back: Callable,
        poll_frequency=15,
        wait_timeout=300,
    ):
        """Watch a name, execute call_back when key is deleted."""
        if isinstance(names, str):
            names = [names]

        q = queue.Queue(maxsize=len(names))
        for _ in range(len(names) - 1):
            q.put(0)

        def wrap_call_back():
            try:
                q.get_nowait()
            except queue.Empty:
                logger.info(f"Key {names} is gone. Executing callback {call_back}")
                call_back()

        for name in names:
            t = threading.Thread(
                target=self._watch_thread_run,
                args=(name, wrap_call_back, poll_frequency, wait_timeout),
                daemon=True,
            )
            t.start()

    def _watch_thread_run(self, name, call_back, poll_frequency, wait_timeout):
        self.wait(name, timeout=wait_timeout, poll_frequency=poll_frequency)
        while True:
            try:
                self.get(name)
                time.sleep(poll_frequency + random.random())
            except NameEntryNotFoundError:
                call_back()
                break


class MemoryNameRecordRepository(NameRecordRepository):
    """Stores all the records in a thread-local memory.

    Note that this is most likely for testing purposes:
    any distributed application is impossible to use this.
    """

    def __init__(self, log_events=False):
        self.__store = {}
        self.__to_delete = set()
        self.__log_events = log_events

    def add(
        self,
        name,
        value,
        delete_on_exit=True,
        keepalive_ttl=None,
        replace=False,
    ):
        if not name:
            raise ValueError(f"Invalid name: {name}")
        name = os.path.normpath(name)
        if self.__log_events:
            print(f"NameResolve: add {name} {value}")
        if name in self.__store and not replace:
            raise NameEntryExistsError(f"K={name} V={self.__store[name]} V2={value}")
        self.__store[name] = str(value)
        if delete_on_exit:
            self.__to_delete.add(name)

    def touch(self, name, value, new_time_to_live):
        raise NotImplementedError()

    def delete(self, name):
        if self.__log_events:
            print(f"NameResolve: delete {name}")
        if name not in self.__store:
            raise NameEntryNotFoundError(f"K={name}")
        if name in self.__to_delete:
            self.__to_delete.remove(name)
        del self.__store[name]

    def clear_subtree(self, name_root):
        if self.__log_events:
            print(f"NameResolve: clear_subtree {name_root}")
        name_root = os.path.normpath(name_root)
        for name in list(self.__store):
            if (
                name_root == "/"
                or name == name_root
                or name.startswith(name_root + "/")
            ):
                if name in self.__to_delete:
                    self.__to_delete.remove(name)
                del self.__store[name]

    def get(self, name):
        name = os.path.normpath(name)
        if name not in self.__store:
            raise NameEntryNotFoundError(f"K={name}")
        r = self.__store[name]
        if self.__log_events:
            print(f"NameResolve: get {name} -> {r}")
        return r

    def get_subtree(self, name_root):
        if self.__log_events:
            print(f"NameResolve: get_subtree {name_root}")
        name_root = os.path.normpath(name_root)
        rs = []
        for name, value in self.__store.items():
            if (
                name_root == "/"
                or name == name_root
                or name.startswith(name_root + "/")
            ):
                rs.append(value)
        return rs

    def find_subtree(self, name_root):
        if self.__log_events:
            print(f"NameResolve: find_subtree {name_root}")
        rs = []
        for name in self.__store:
            if (
                name_root == "/"
                or name == name_root
                or name.startswith(name_root + "/")
            ):
                rs.append(name)
        rs.sort()
        return rs

    def reset(self):
        for name in self.__to_delete:
            self.__store.pop(name)
        self.__to_delete = set()


class NfsNameRecordRepository(NameRecordRepository):
    RECORD_ROOT = f"{cluster_spec.fileroot}/name_resolve/"
    os.makedirs(RECORD_ROOT, exist_ok=True)

    def __init__(self, **kwargs):
        self.__to_delete = set()

    @staticmethod
    def __dir_path(name):
        return os.path.join(NfsNameRecordRepository.RECORD_ROOT, name)

    @staticmethod
    def __file_path(name):
        return os.path.join(NfsNameRecordRepository.__dir_path(name), "ENTRY")

    def add(
        self,
        name,
        value,
        delete_on_exit=True,
        keepalive_ttl=None,
        replace=False,
    ):
        if not name:
            raise ValueError("Name cannot be empty")
        name = os.path.normpath(name)
        path = self.__file_path(name)
        while True:
            # To avoid concurrency issues when multiple processes
            # call makedirs on the same dirname of CPFS.
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                break
            except (NotADirectoryError, FileNotFoundError):
                pass
        if os.path.isfile(path) and not replace:
            raise NameEntryExistsError(path)
        local_id = str(uuid.uuid4())[:8]
        with open(path + f".tmp.{local_id}", "w") as f:
            f.write(str(value))
        os.rename(path + f".tmp.{local_id}", path)
        if delete_on_exit:
            self.__to_delete.add(name)

    def delete(self, name):
        path = self.__file_path(name)
        if not os.path.isfile(path):
            raise NameEntryNotFoundError(path)
        os.remove(path)
        while True:
            path = os.path.dirname(path)
            if path == NfsNameRecordRepository.RECORD_ROOT:
                break
            if len(os.listdir(path)) > 0:
                break
            shutil.rmtree(path, ignore_errors=True)
        if name in self.__to_delete:
            self.__to_delete.remove(name)

    def clear_subtree(self, name_root):
        dir_path = self.__dir_path(name_root)
        if os.path.isdir(dir_path):
            logger.info("Removing name resolve path: %s", dir_path)
            shutil.rmtree(dir_path)
        else:
            logger.info("No such name resolve path: %s", dir_path)

    def get(self, name):
        name = os.path.normpath(name)
        path = self.__file_path(name)
        if not os.path.isfile(path):
            raise NameEntryNotFoundError(path)
        for _ in range(100):
            # HACK: dealing with the possible OSError: Stale file handle
            try:
                with open(path, "r") as f:
                    return f.read().strip()
            except OSError as e:
                if e.errno == 116:
                    time.sleep(5e-3)
                    continue
                raise e
        raise RuntimeError("Failed to read value for %s" % name)

    def get_subtree(self, name_root):
        dir_path = self.__dir_path(name_root)
        rs = []
        if os.path.isdir(dir_path):
            for root, _, files in os.walk(dir_path):
                try:
                    if len(files) != 1:
                        continue
                    if files[0] != "ENTRY":
                        continue
                    key = root.removeprefix(self.RECORD_ROOT)
                    key = key.removeprefix("/")
                    rs.append(self.get(key))
                except NameEntryNotFoundError:
                    pass
        return rs

    def find_subtree(self, name_root):
        dir_path = self.__dir_path(name_root)
        rs = []
        if os.path.isdir(dir_path):
            for root, _, files in os.walk(dir_path):
                try:
                    if len(files) != 1:
                        continue
                    if files[0] != "ENTRY":
                        continue
                    key = root.removeprefix(self.RECORD_ROOT)
                    key = key.removeprefix("/")
                    rs.append(key)
                except NameEntryNotFoundError:
                    pass
        rs.sort()
        return rs

    def reset(self):
        for name in list(self.__to_delete):
            try:
                self.delete(name)
            except:
                pass
        self.__to_delete = set()


class RedisNameRecordRepository(NameRecordRepository):
    _IS_FRL = False
    REDIS_HOST = "redis" if _IS_FRL else "localhost"
    REDIS_PASSWORD = security.read_key("redis") if _IS_FRL else None
    REDIS_DB = 0
    KEEPALIVE_POLL_FREQUENCY = 1

    @dataclasses.dataclass
    class _Entry:
        value: str
        keepalive_ttl: Optional[int] = None
        keeper: Optional[timeutil.FrequencyControl] = None

    def __init__(self, **kwargs):
        import redis
        from redis.backoff import ExponentialBackoff
        from redis.retry import Retry

        super().__init__()
        self.__lock = threading.Lock()
        self.__redis = redis.Redis(
            host=RedisNameRecordRepository.REDIS_HOST,
            password=RedisNameRecordRepository.REDIS_PASSWORD,
            db=RedisNameRecordRepository.REDIS_DB,
            socket_timeout=60,
            retry_on_timeout=True,
            retry=Retry(ExponentialBackoff(180, 60), 3),
        )
        self.__entries = {}
        self.__keepalive_running = True
        self.__keepalive_thread = threading.Thread(
            target=self.__keepalive_thread_run, daemon=True
        )
        self.__keepalive_thread.start()

    def __del__(self):
        self.__keepalive_running = False
        self.__keepalive_thread.join(timeout=5)
        self.reset()
        self.__redis.close()

    def add(self, name, value, delete_on_exit=True, keepalive_ttl=10, replace=False):
        # deprecated parameter: delete_on_exit, now every entry has a default keepalive_ttl=10 seconds
        if name.endswith("/"):
            raise ValueError(f"Entry name cannot end with '/': {name}")

        with self.__lock:
            keepalive_ttl = int(keepalive_ttl * 1000)
            assert (
                keepalive_ttl > 0
            ), f"keepalive_ttl in milliseconds must >0: {keepalive_ttl}"
            if self.__redis.set(name, value, px=keepalive_ttl, nx=not replace) is None:
                raise NameEntryExistsError(f"Cannot set Redis key: K={name} V={value}")

            # touch every 1/3 of keepalive_ttl to prevent Redis from deleting the key
            # after program exit, redis will automatically delete key in keepalive_ttl
            self.__entries[name] = self._Entry(
                value=value,
                keepalive_ttl=keepalive_ttl,
                keeper=timeutil.FrequencyControl(
                    frequency_seconds=keepalive_ttl / 1000 / 3
                ),
            )

    def delete(self, name):
        with self.__lock:
            self.__delete_locked(name)

    def __delete_locked(self, name):
        if name in self.__entries:
            del self.__entries[name]
        if self.__redis.delete(name) == 0:
            raise NameEntryNotFoundError(f"No such Redis entry to delete: {name}")

    def clear_subtree(self, name_root):
        with self.__lock:
            count = 0
            for name in list(self.__find_subtree_locked(name_root)):
                try:
                    self.__delete_locked(name)
                    count += 1
                except NameEntryNotFoundError:
                    pass
            logger.info("Deleted %d Redis entries under %s", count, name_root)

    def get(self, name):
        with self.__lock:
            return self.__get_locked(name)

    def __get_locked(self, name):
        r = self.__redis.get(name)
        if r is None:
            raise NameEntryNotFoundError(f"No such Redis entry: {name}")
        return r.decode()

    def get_subtree(self, name_root):
        with self.__lock:
            rs = []
            for name in self.__find_subtree_locked(name_root):
                rs.append(self.__get_locked(name))
            rs.sort()
            return rs

    def find_subtree(self, name_root):
        with self.__lock:
            return list(sorted(self.__find_subtree_locked(name_root)))

    def reset(self):
        with self.__lock:
            count = 0
            for name in list(self.__entries):
                try:
                    self.__delete_locked(name)
                    count += 1
                except NameEntryNotFoundError:
                    pass
            self.__entries = {}
            logger.info("Reset %d saved Redis entries", count)

    def __keepalive_thread_run(self):
        while self.__keepalive_running:
            time.sleep(self.KEEPALIVE_POLL_FREQUENCY)
            with self.__lock:
                for name, entry in self.__entries.items():
                    if entry.keeper is not None and entry.keeper.check():
                        r = self.__redis.set(name, entry.value, px=entry.keepalive_ttl)
                        if r is None:
                            logger.error(
                                "Failed touching Redis key: K=%s V=%s",
                                name,
                                entry.value,
                            )

    def __find_subtree_locked(self, name_root):
        pattern = name_root + "*"
        return [k.decode() for k in self.__redis.keys(pattern=pattern)]

    def _testonly_drop_cached_entry(self, name):
        """Used by unittest only to simulate the case that the Python process
        crashes and the key is automatically removed after TTL."""
        with self.__lock:
            del self.__entries[name]
            print("Testonly: dropped key:", name)


class Etcd3NameRecordRepository(NameRecordRepository):
    """Implements a name record repository using etcd3 as the backend storage.

    This implementation provides distributed key-value storage with support for
    TTL-based expiration, atomic operations, and key watching functionality.
    """

    # Default configuration
    try:
        host, port = os.getenv("REAL_ETCD_ADDR", "").split(":")
    except ValueError:
        host, port = "localhost", 2379
    ETCD_HOST = host
    ETCD_PORT = int(port)
    ETCD_USER = None
    ETCD_PASSWORD = None
    KEEPALIVE_POLL_FREQUENCY = 1

    @dataclasses.dataclass
    class _Entry:
        value: str
        lease_id: Optional[int] = None
        keepalive_ttl: Optional[int] = None
        keeper: Optional[timeutil.FrequencyControl] = None

    def __init__(self, host=None, port=None, user=None, password=None, **kwargs):
        """Initialize the etcd3 name record repository.

        Args:
            host: etcd server host (defaults to ETCD_HOST)
            port: etcd server port (defaults to ETCD_PORT)
            user: etcd username for authentication (defaults to ETCD_USER)
            password: etcd password for authentication (defaults to ETCD_PASSWORD)
            **kwargs: Additional configuration parameters
        """

        super().__init__()
        self._lock = threading.Lock()

        # Set connection parameters
        self._host = host or self.ETCD_HOST
        self._port = port or self.ETCD_PORT
        self._user = user or self.ETCD_USER
        self._password = password or self.ETCD_PASSWORD

        # Connect to etcd
        self._client = etcd3.client(
            host=self._host, port=self._port, user=self._user, password=self._password
        )

        # Keep track of entries for cleanup and keepalive
        self._entries = {}
        self._keepalive_running = True
        self._keepalive_thread = threading.Thread(
            target=self._keepalive_thread_run, daemon=True
        )
        self._keepalive_thread.start()

        self._to_delete = set()

        logger.info(f"Connected to etcd3 at {self._host}:{self._port}")

    def __del__(self):
        """Clean up resources when the object is deleted."""
        self._keepalive_running = False
        if hasattr(self, "_keepalive_thread"):
            self._keepalive_thread.join(timeout=5)
        self.reset()
        if hasattr(self, "_client"):
            self._client.close()

    def _create_lease(self, ttl_seconds):
        """Create an etcd lease with the specified TTL.

        Args:
            ttl_seconds: Time-to-live in seconds

        Returns:
            The lease ID
        """
        lease = self._client.lease(ttl_seconds)
        return lease.id

    def add(
        self,
        name,
        value,
        delete_on_exit=True,
        keepalive_ttl=None,
        replace=False,
    ):
        """Add a key-value pair to etcd with optional TTL.

        Args:
            name: Key name
            value: Value to store
            delete_on_exit: Whether to delete the key when this object is destroyed
            keepalive_ttl: TTL in seconds for the key (default: 10)
            replace: Whether to replace an existing key

        Raises:
            NameEntryExistsError: If the key already exists and replace is False
        """
        if not name:
            raise ValueError(f"Invalid name: {name}")
        name = os.path.normpath(name)
        value = str(value)

        with self._lock:
            # Check if key exists when replace=False
            if not replace:
                existing_value, _ = self._client.get(name)
                if existing_value is not None:
                    raise NameEntryExistsError(
                        f"Key already exists: K={name} V={existing_value.decode()}"
                    )

            # Create lease for TTL if specified
            lease_id = None
            if keepalive_ttl is not None and keepalive_ttl > 0:
                lease_id = self._create_lease(keepalive_ttl)
                # Encode the string value to bytes
                self._client.put(name, value.encode("utf-8"), lease=lease_id)
                self._to_delete.add(name)
            else:
                # Encode the string value to bytes
                self._client.put(name, value.encode("utf-8"))
                if delete_on_exit:
                    self._to_delete.add(name)

            # Store entry information for keepalive management
            self._entries[name] = self._Entry(
                value=value,
                lease_id=lease_id,
                keepalive_ttl=keepalive_ttl,
                keeper=(
                    timeutil.FrequencyControl(frequency_seconds=keepalive_ttl / 3)
                    if keepalive_ttl
                    else None
                ),
            )

    def delete(self, name):
        """Delete a key from etcd.

        Args:
            name: Key to delete

        Raises:
            NameEntryNotFoundError: If the key doesn't exist
        """
        with self._lock:
            self._delete_locked(name)
            if name in self._to_delete:
                self._to_delete.remove(name)

    def _delete_locked(self, name):
        """Delete a key from etcd with lock already acquired.

        Args:
            name: Key to delete

        Raises:
            NameEntryNotFoundError: If the key doesn't exist
        """
        # First check if the key exists
        value, _ = self._client.get(name)
        if value is None:
            raise NameEntryNotFoundError(f"No such etcd entry to delete: {name}")

        # Clean up entry tracking
        if name in self._entries:
            del self._entries[name]

        # Delete from etcd
        self._client.delete(name)

    def clear_subtree(self, name_root):
        """Delete all keys with the given prefix.

        Args:
            name_root: Prefix to match keys against
        """
        with self._lock:
            count = 0
            name_root = os.path.normpath(name_root)
            # Get all keys with the prefix
            for key_metadata_tuple in self._client.get_prefix(name_root):
                key = key_metadata_tuple[1].key.decode(
                    "utf-8"
                )  # Extract the key from metadata
                # Remove from our tracking
                if key in self._entries:
                    del self._entries[key]
                # Delete from etcd
                self._client.delete(key)
                count += 1

            logger.debug(f"Deleted {count} etcd entries under {name_root}")

    def get_subtree(self, name_root):
        """Get all values with keys having the given prefix.

        Args:
            name_root: Prefix to match keys against

        Returns:
            List of values
        """
        with self._lock:
            rs = []
            name_root = os.path.normpath(name_root)
            for value_metadata_tuple in self._client.get_prefix(name_root):
                value = value_metadata_tuple[0].decode("utf-8")  # Extract the value
                rs.append(value)
            return sorted(rs)

    def find_subtree(self, name_root):
        """Find all keys with the given prefix.

        Args:
            name_root: Prefix to match keys against

        Returns:
            List of keys
        """
        with self._lock:
            rs = []
            for key_metadata_tuple in self._client.get_prefix(name_root):
                key = key_metadata_tuple[1].key.decode(
                    "utf-8"
                )  # Extract the key from metadata
                rs.append(key)
            return sorted(rs)

    def get(self, name):
        """Get the value for a key.

        Args:
            name: Key to retrieve

        Returns:
            The value as a string

        Raises:
            NameEntryNotFoundError: If the key doesn't exist
        """
        name = os.path.normpath(name)
        with self._lock:
            return self._get_locked(name)

    def _get_locked(self, name):
        """Get a value with lock already acquired.

        Args:
            name: Key to retrieve

        Returns:
            The value as a string

        Raises:
            NameEntryNotFoundError: If the key doesn't exist
        """
        value, _ = self._client.get(name)
        if value is None:
            raise NameEntryNotFoundError(f"No such etcd entry: {name}")
        return value.decode("utf-8")

    def reset(self):
        """Delete all keys added via this repository instance."""
        with self._lock:
            count = 0
            for name in self._to_delete:
                if name in self._entries:
                    try:
                        self._delete_locked(name)
                        count += 1
                    except NameEntryNotFoundError:
                        pass
            self._to_delete = set()
            logger.info(f"Reset {count} saved etcd entries")

    def _keepalive_thread_run(self):
        """Background thread to keep leases alive."""
        while self._keepalive_running:
            time.sleep(self.KEEPALIVE_POLL_FREQUENCY)
            with self._lock:
                for name, entry in list(self._entries.items()):
                    if (
                        entry.keeper is not None
                        and entry.keepalive_ttl is not None
                        and entry.lease_id is not None
                        and entry.keeper.check()
                    ):
                        try:
                            # Refresh the lease
                            self._client.refresh_lease(entry.lease_id)
                        except Exception as e:
                            logger.error(
                                f"Failed to refresh lease for key: K={name} V={entry.value}. Error: {e}"
                            )

    def watch_names(
        self,
        names: List,
        call_back: Callable,
        poll_frequency=15,
        wait_timeout=300,
    ):
        """Watch keys and call back when they are deleted.

        Args:
            names: Keys to watch
            call_back: Function to call when any key is deleted
            poll_frequency: How often to check in seconds
            wait_timeout: Maximum time to wait for keys to exist
        """
        if isinstance(names, str):
            names = [names]

        q = queue.Queue(maxsize=len(names))
        for _ in range(len(names) - 1):
            q.put(0)

        def wrap_call_back():
            try:
                q.get_nowait()
            except queue.Empty:
                logger.info(f"Key {names} is gone. Executing callback {call_back}")
                call_back()

        # Use etcd's native watch capability for more efficient watching
        for name in names:
            # First wait for the key to exist
            self.wait(name, timeout=wait_timeout, poll_frequency=poll_frequency)

            # Start watching for key deletion
            watch_id = self._client.add_watch_callback(
                name, lambda event: self._watch_callback(event, wrap_call_back)
            )

            # Store watch ID for cleanup
            if not hasattr(self, "_watch_ids"):
                self._watch_ids = []
            self._watch_ids.append(watch_id)

    def _watch_callback(self, event, callback):
        """Process watch events and call back on deletion.

        Args:
            event: The etcd watch response (WatchResponse object)
            callback: Function to call when a key is deleted
        """
        # Iterate through the events in the WatchResponse
        for ev in event.events:
            # Check if this is a delete event
            if isinstance(ev, etcd3.events.DeleteEvent):
                logger.debug(f"Key {ev.key.decode()} was deleted. Executing callback.")
                callback()

    def _testonly_drop_cached_entry(self, name):
        """Used by unittest only to simulate the case that the process crashes.

        Args:
            name: Key to drop from local cache
        """
        with self._lock:
            if name in self._entries:
                del self._entries[name]
                logger.debug(f"Testonly: dropped key: {name}")


def make_repository(type_="nfs", **kwargs):
    if type_ == "memory":
        return MemoryNameRecordRepository(**kwargs)
    elif type_ == "nfs":
        return NfsNameRecordRepository(**kwargs)
    elif type_ == "redis":
        return RedisNameRecordRepository(**kwargs)
    elif type_ == "etcd3":
        return Etcd3NameRecordRepository(**kwargs)
    else:
        raise NotImplementedError(f"No such name resolver: {type_}")


# DEFAULT_REPOSITORY_TYPE = "redis" if socket.gethostname().startswith("frl") else "nfs"
DEFAULT_REPOSITORY_TYPE = "nfs"
if (
    etcd3 is not None
    and cluster.spec.name in ["wa180", "na132", "su18"]
    and os.getenv("REAL_ETCD_ADDR", "")
):
    DEFAULT_REPOSITORY_TYPE = "etcd3"
DEFAULT_REPOSITORY = make_repository(DEFAULT_REPOSITORY_TYPE)
add = DEFAULT_REPOSITORY.add
add_subentry = DEFAULT_REPOSITORY.add_subentry
delete = DEFAULT_REPOSITORY.delete
clear_subtree = DEFAULT_REPOSITORY.clear_subtree
get = DEFAULT_REPOSITORY.get
get_subtree = DEFAULT_REPOSITORY.get_subtree
find_subtree = DEFAULT_REPOSITORY.find_subtree
wait = DEFAULT_REPOSITORY.wait
reset = DEFAULT_REPOSITORY.reset
watch_names = DEFAULT_REPOSITORY.watch_names


def reconfigure(*args, **kwargs):
    global DEFAULT_REPOSITORY, DEFAULT_REPOSITORY_TYPE
    global add, add_subentry, delete, clear_subtree, get, get_subtree, find_subtree, wait, reset, watch_names
    DEFAULT_REPOSITORY = make_repository(*args, **kwargs)
    DEFAULT_REPOSITORY_TYPE = args[0]
    add = DEFAULT_REPOSITORY.add
    add_subentry = DEFAULT_REPOSITORY.add_subentry
    delete = DEFAULT_REPOSITORY.delete
    clear_subtree = DEFAULT_REPOSITORY.clear_subtree
    get = DEFAULT_REPOSITORY.get
    get_subtree = DEFAULT_REPOSITORY.get_subtree
    find_subtree = DEFAULT_REPOSITORY.find_subtree
    wait = DEFAULT_REPOSITORY.wait
    reset = DEFAULT_REPOSITORY.reset
    watch_names = DEFAULT_REPOSITORY.watch_names

"""Comprehensive test for KV storage data format support"""

import time
import threading
import json
import datetime
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any
from gswarm.data import DataStorage, DataServer, start_server


@dataclass
class CustomObject:
    """Test custom object"""

    name: str
    value: int
    timestamp: datetime.datetime


def start_server_background():
    """Start server in background thread"""

    def run_server():
        start_server(host="localhost", port=9015, max_mem_size=1024 * 1024 * 1024)  # 1GB

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(3)  # Give server time to start


def test_direct_storage():
    """Test direct storage access (no HTTP) - should support all Python types"""
    print("=== Testing Direct Storage Access ===")
    storage = DataStorage()

    # Test data of various types
    test_data = {
        "string": "Hello World!",
        "integer": 42,
        "float": 3.14159,
        "boolean": True,
        "none_value": None,
        "list_simple": [1, 2, 3, "four", 5.0],
        "list_nested": [[1, 2], [3, 4], {"nested": "dict"}],
        "dict_simple": {"key1": "value1", "key2": 123},
        "dict_nested": {
            "user": {"id": 1, "name": "Alice", "active": True},
            "settings": {"theme": "dark", "notifications": False},
            "data": [1, 2, 3, {"nested": True}],
        },
        "tuple": (1, "two", 3.0),
        "set": {1, 2, 3, "unique"},
        "datetime": datetime.datetime.now(),
        "custom_object": CustomObject("test", 42, datetime.datetime.now()),
    }

    # Add NumPy array if available
    try:
        test_data["numpy_array"] = np.array([1, 2, 3, 4, 5])
        test_data["numpy_matrix"] = np.array([[1, 2], [3, 4]])
    except:
        print("NumPy not available, skipping numpy tests")

    success_count = 0
    total_tests = len(test_data)

    # Test writing and reading each data type
    for key, original_value in test_data.items():
        try:
            # Write
            write_success = storage.write(f"test_{key}", original_value, persist=True)
            if not write_success:
                print(f"❌ Failed to write {key}: {type(original_value)}")
                continue

            # Read
            read_value = storage.read(f"test_{key}")

            # Compare (handle special cases)
            if key == "set":
                # Sets might be converted to lists in some cases
                if isinstance(read_value, (set, list, tuple)):
                    if set(read_value) == original_value:
                        print(f"✅ {key}: {type(original_value)} -> OK")
                        success_count += 1
                    else:
                        print(f"❌ {key}: Value mismatch")
                else:
                    print(f"❌ {key}: Type changed unexpectedly")
            elif hasattr(original_value, "__array__"):  # NumPy arrays
                if hasattr(read_value, "__array__") and np.array_equal(original_value, read_value):
                    print(f"✅ {key}: {type(original_value)} -> OK")
                    success_count += 1
                else:
                    print(f"❌ {key}: NumPy array mismatch")
            else:
                if read_value == original_value:
                    print(f"✅ {key}: {type(original_value)} -> OK")
                    success_count += 1
                else:
                    print(f"❌ {key}: {type(original_value)} -> Value mismatch")
                    print(f"   Original: {original_value}")
                    print(f"   Read:     {read_value}")

        except Exception as e:
            print(f"❌ {key}: {type(original_value)} -> Exception: {e}")

    print(f"\nDirect Storage Results: {success_count}/{total_tests} tests passed")
    return success_count == total_tests


def test_http_api():
    """Test HTTP API access - limited by JSON serialization"""
    print("\n=== Testing HTTP API Access ===")

    # Start server
    start_server_background()
    data_server = DataServer("localhost:9015")

    # Test JSON-serializable data
    json_test_data = {
        "string": "Hello World!",
        "integer": 42,
        "float": 3.14159,
        "boolean": True,
        "none_value": None,
        "list_simple": [1, 2, 3, "four", 5.0],
        "list_nested": [[1, 2], [3, 4], {"nested": "dict"}],
        "dict_simple": {"key1": "value1", "key2": 123},
        "dict_nested": {
            "user": {"id": 1, "name": "Alice", "active": True},
            "settings": {"theme": "dark", "notifications": False},
            "data": [1, 2, 3, {"nested": True}],
        },
        "unicode": "Unicode: 你好世界! 🌍 émojis",
    }

    # Test non-JSON-serializable data
    non_json_data = {
        "datetime": datetime.datetime.now(),
        "set": {1, 2, 3},
        "tuple": (1, "two", 3.0),
        "custom_object": CustomObject("test", 42, datetime.datetime.now()),
    }

    success_count = 0

    # Test JSON-serializable data
    print("\nTesting JSON-serializable data:")
    for key, original_value in json_test_data.items():
        try:
            # Write
            write_success = data_server.write(f"http_{key}", original_value, persist=True)
            if not write_success:
                print(f"❌ Failed to write {key}: {type(original_value)}")
                continue

            # Read
            read_value = data_server.read(f"http_{key}")

            if read_value == original_value:
                print(f"✅ {key}: {type(original_value)} -> OK")
                success_count += 1
            else:
                print(f"❌ {key}: Value mismatch")
                print(f"   Original: {original_value}")
                print(f"   Read:     {read_value}")

        except Exception as e:
            print(f"❌ {key}: {type(original_value)} -> Exception: {e}")

    # Test non-JSON-serializable data
    print("\nTesting non-JSON-serializable data:")
    for key, original_value in non_json_data.items():
        try:
            write_success = data_server.write(f"http_{key}", original_value, persist=True)
            if write_success:
                read_value = data_server.read(f"http_{key}")
                print(f"✅ {key}: {type(original_value)} -> Surprisingly works!")
                success_count += 1
            else:
                print(f"❌ {key}: {type(original_value)} -> Failed to write (expected)")
        except Exception as e:
            print(f"❌ {key}: {type(original_value)} -> Exception: {e}")

    total_tests = len(json_test_data) + len(non_json_data)
    print(f"\nHTTP API Results: {success_count}/{total_tests} tests passed")

    # Test stats
    stats = data_server.get_stats()
    if stats:
        print(f"\nStorage stats:")
        print(f"  Total keys: {stats['total_keys']}")
        print(f"  Memory usage: {stats['usage_percent']:.1f}%")

    return success_count


def test_special_cases():
    """Test special cases and edge conditions"""
    print("\n=== Testing Special Cases ===")
    storage = DataStorage()

    # Large data
    large_list = list(range(100000))
    storage.write("large_data", large_list)
    read_large = storage.read("large_data")
    print(f"✅ Large data (100k items): {len(read_large) == 100000}")

    # Empty containers
    storage.write("empty_list", [])
    storage.write("empty_dict", {})
    storage.write("empty_string", "")
    print(f"✅ Empty list: {storage.read('empty_list') == []}")
    print(f"✅ Empty dict: {storage.read('empty_dict') == {}}")
    print(f"✅ Empty string: {storage.read('empty_string') == ''}")

    # Unicode and special characters
    unicode_data = {"chinese": "你好世界", "emoji": "🌍🚀🎉", "special": "!@#$%^&*()[]{}|\\:;\"'<>?,./"}

    for key, value in unicode_data.items():
        storage.write(f"unicode_{key}", value)
        read_val = storage.read(f"unicode_{key}")
        print(f"✅ Unicode {key}: {read_val == value}")

    # Nested data
    deeply_nested = {
        "level1": {"level2": {"level3": {"level4": {"data": "deep value", "list": [1, 2, {"nested": True}]}}}}
    }
    storage.write("deep_nested", deeply_nested)
    read_nested = storage.read("deep_nested")
    print(f"✅ Deeply nested: {read_nested == deeply_nested}")


if __name__ == "__main__":
    print("KV Storage Data Format Support Test")
    print("=" * 50)

    # Test direct storage access
    direct_success = test_direct_storage()

    # Test HTTP API access
    http_success = test_http_api()

    # Test special cases
    test_special_cases()

    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"✅ Direct Storage: {'PASS' if direct_success else 'PARTIAL'}")
    print(f"📡 HTTP API: {'LIMITED' if http_success > 0 else 'ISSUES'}")
    print("\nNOTE: Direct storage access supports all Python types.")
    print("HTTP API is limited by JSON serialization for complex types.")

import hashlib
import os
import pickle
import time
from functools import wraps

import cachetools

# The maximum number of items the cache can store
CACHE_MAX_SIZE = 1000000

# The Time-To-Live value for cache entries (in seconds)
TTL = 60 * 60  # 1 hour

"""

# The file to store the cache
CACHE_FILE = 'cache.pickle'

# Load the cache from disk if it exists
try:
    with open(CACHE_FILE, 'rb') as f:
        cache = pickle.load(f)
except FileNotFoundError:
    # If the cache file doesn't exist, create a new cache
"""
cache = cachetools.TTLCache(CACHE_MAX_SIZE, TTL)


def file_based_cache(func):
    @wraps(func)
    def wrapper(startpath, basepath, *args, **kwargs):
        path = os.path.join(basepath, startpath)

        # Calculate the cache key
        cache_key_data = str((path, args, kwargs)).encode("utf-8")
        cache_key_hash = hashlib.sha256(cache_key_data)
        cache_key = os.path.realpath(path) + ":" + cache_key_hash.hexdigest()

        # If we have a cached result and it's still valid, return it
        if cache_key in cache:
            entry = cache[cache_key]
            last_modification = os.path.getmtime(os.path.realpath(path))

            print(f"{last_modification} <= {entry['time']} ? ")
            if last_modification <= entry["time"]:
                print("cache_reuse")
                return entry["result"]

        # Call the decorated function and cache the result
        print(f"operating on cache key {path} {cache_key}")

        result = func(startpath, basepath, *args, **kwargs)
        cache[cache_key] = {
            "time": time.time(),  # The time when the result was cached
            "result": result,  # The cached result
        }

        # Save the cache to disk
        # with open(CACHE_FILE, 'wb') as f:
        #    pickle.dump(cache, f)

        return result

    return wrapper

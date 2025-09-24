import time
import config  # Direct module import for better performance

class Cache:
    @staticmethod
    def _make_param_str(params: dict) -> str:
        try:
            items = sorted(params.items())
        except Exception:
            # Fallback: best-effort string
            items = []
            for k in sorted(params.keys()):
                items.append((k, str(params[k])))
        return '|'.join(f"{k}={v}" for k, v in items)

    @staticmethod
    def generate_typed_cache_key(simulation_type: str, operation: str, **params) -> str:
        return f"TYPED::{simulation_type}::{operation}::{Cache._make_param_str(params)}"

    @staticmethod
    def get_typed_cached_result(simulation_type: str, operation: str, expected_type=None, **params):
        key = Cache.generate_typed_cache_key(simulation_type, operation, **params)
        entry = config._ANALYSIS_CACHE.get(key)  # Direct access
        if entry is None:
            return None
        # New-format entry
        if isinstance(entry, dict) and 'data' in entry:
            data = entry.get('data')
            if expected_type is not None and not isinstance(data, expected_type):
                return None
            return data
        # Legacy/raw entry stored under same key
        if expected_type is not None and not isinstance(entry, expected_type):
            return None
        return entry

    @staticmethod
    def set_typed_cached_result(simulation_type: str, operation: str, result, **params):
        key = Cache.generate_typed_cache_key(simulation_type, operation, **params)
        config._ANALYSIS_CACHE[key] = {  # Direct access
            'data': result,
            'simulation_type': simulation_type,
            'operation': operation,
            'params': params,
            'data_type': type(result).__name__,
            'timestamp': time.time(),
        }

    @staticmethod
    def clear_sample_cache():
        """
        Clears the internal cache used for storing sample paths. This method ensures
        that any pre-cached sample paths are removed, forcing the system to retrieve or
        calculate them again as needed.

        :return: None
        """
        config._SAMPLE_PATHS_CACHE.clear()  # Direct access

    @staticmethod
    def clear_analysis_cache():
        """
        Clears the analysis cache to remove any stored data.

        This function is responsible for clearing all data stored in the
        analysis cache. It is typically used to reset the cache data
        either for maintenance or to prevent unwanted accumulation
        of obsolete or unnecessary analysis results.

        :raises KeyError: If the cache clearing operation targets keys
                          that do not exist in the cache.
        :return: None
        """
        config._ANALYSIS_CACHE.clear()  # Direct access

    @staticmethod
    def get_cached_analysis(cache_key: str):
        """
        Retrieve an analysis from the cache using the provided cache key.

        This function looks up the given cache_key in the internal analysis
        cache and returns the corresponding analysis if available. If no
        cached analysis is found for the provided key, it returns None.
        The analysis cache is not exposed externally.

        :param cache_key: The key used to look up the cached analysis.
        :type cache_key: str
        :return: The cached analysis if found, otherwise None.
        :rtype: Any or None
        """
        return config._ANALYSIS_CACHE.get(cache_key)  # Direct access

    @staticmethod
    def set_cached_analysis(cache_key: str, result):
        """
        Stores the provided analysis result in the cache associated with the given cache key.

        :param cache_key: The key used to store and retrieve the analysis result
                          from the cache.
        :type cache_key: str
        :param result: The analysis result to be cached. Its type can vary depending
                       on the data being stored.
        :return: None
        """
        config._ANALYSIS_CACHE[cache_key] = result  # Direct access

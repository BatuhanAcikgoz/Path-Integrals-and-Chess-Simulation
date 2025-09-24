"""
Performans optimizasyonu modülü
Bu modül bellek yönetimi, önbellekleme ve performans izleme fonksiyonları içerir.
"""

import gc
import os
import time
import functools
from typing import Any, Dict, List, Optional, Callable
import json
import pickle

class PerformanceOptimizer:
    """Performans optimizasyonu sınıfı"""

    def __init__(self, cache_dir: str = "cache", max_cache_size: int = 1000):
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size
        self.memory_cache: Dict[str, Any] = {}
        self.call_stats: Dict[str, Dict] = {}

        # Cache dizinini oluştur
        os.makedirs(cache_dir, exist_ok=True)

    def monitor_memory(self) -> float:
        """Bellek kullanımını MB cinsinden döndürür"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0

    def disk_cache(self, cache_key: Optional[str] = None, expire_hours: int = 24):
        """Disk tabanlı önbellekleme decorator'ü"""
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Cache key oluştur
                key = cache_key or f"{func.__name__}_{hash(str(args) + str(kwargs))}"
                cache_file = os.path.join(self.cache_dir, f"{key}.pkl")

                # Cache kontrolü
                if os.path.exists(cache_file):
                    try:
                        # Dosya yaşını kontrol et
                        file_age = time.time() - os.path.getmtime(cache_file)
                        if file_age < expire_hours * 3600:
                            with open(cache_file, 'rb') as f:
                                result = pickle.load(f)
                                print(f"⚡ Cache hit for {func.__name__}")
                                return result
                    except Exception as e:
                        print(f"⚠️ Cache read error: {e}")

                # Fonksiyonu çalıştır ve sonucu kaydet
                result = func(*args, **kwargs)

                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(result, f)
                except Exception as e:
                    print(f"⚠️ Cache write error: {e}")

                return result
            return wrapper
        return decorator

    def memory_cache_decorator(self, max_size: int = 100):
        """Bellek tabanlı önbellekleme decorator'ü"""
        def decorator(func: Callable):
            cache = {}

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                key = str(args) + str(kwargs)

                if key in cache:
                    print(f"⚡ Memory cache hit for {func.__name__}")
                    return cache[key]

                result = func(*args, **kwargs)

                # Cache boyutunu kontrol et
                if len(cache) >= max_size:
                    # LRU eviction - en eski girdiyi sil
                    oldest_key = next(iter(cache))
                    del cache[oldest_key]

                cache[key] = result
                return result
            return wrapper
        return decorator

    def time_function(self, func: Callable):
        """Fonksiyon çalışma süresini ölçer"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = self.monitor_memory()

            result = func(*args, **kwargs)

            end_time = time.time()
            end_memory = self.monitor_memory()

            # İstatistikleri kaydet
            func_name = func.__name__
            if func_name not in self.call_stats:
                self.call_stats[func_name] = {
                    'call_count': 0,
                    'total_time': 0,
                    'avg_time': 0,
                    'memory_usage': []
                }

            stats = self.call_stats[func_name]
            stats['call_count'] += 1
            stats['total_time'] += (end_time - start_time)
            stats['avg_time'] = stats['total_time'] / stats['call_count']
            stats['memory_usage'].append(end_memory - start_memory)

            print(f"⏱️ {func_name}: {end_time - start_time:.3f}s, "
                  f"Memory: {end_memory - start_memory:+.1f}MB")

            return result
        return wrapper

    def force_garbage_collection(self):
        """Agresif bellek temizliği"""
        collected = gc.collect()
        print(f"🧹 Garbage collection: {collected} objects freed")
        return collected

    def clear_cache(self, cache_type: str = "all"):
        """Önbellekleri temizle"""
        if cache_type in ["all", "memory"]:
            self.memory_cache.clear()
            print("🗑️ Memory cache cleared")

        if cache_type in ["all", "disk"]:
            try:
                for file in os.listdir(self.cache_dir):
                    if file.endswith('.pkl'):
                        os.remove(os.path.join(self.cache_dir, file))
                print("🗑️ Disk cache cleared")
            except Exception as e:
                print(f"⚠️ Error clearing disk cache: {e}")

    def get_performance_report(self) -> Dict:
        """Performans raporu oluştur"""
        report = {
            'current_memory_mb': self.monitor_memory(),
            'function_stats': self.call_stats,
            'cache_info': {
                'memory_cache_size': len(self.memory_cache),
                'disk_cache_files': len([f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')])
            }
        }
        return report

    def save_performance_report(self, filepath: str = "performance_report.json"):
        """Performans raporunu dosyaya kaydet"""
        report = self.get_performance_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"📊 Performance report saved to {filepath}")

# Global optimizer instance
optimizer = PerformanceOptimizer()

# Decorator shortcuts
disk_cache = optimizer.disk_cache
memory_cache = optimizer.memory_cache_decorator
time_function = optimizer.time_function

def optimize_pandas_memory(df):
    """Pandas DataFrame bellek kullanımını optimize et"""
    import pandas as pd

    start_memory = df.memory_usage(deep=True).sum() / 1024**2

    # Numeric kolonları optimize et
    for col in df.select_dtypes(include=['int']).columns:
        col_min = df[col].min()
        col_max = df[col].max()

        if col_min >= 0:
            if col_max < 255:
                df[col] = df[col].astype('uint8')
            elif col_max < 65535:
                df[col] = df[col].astype('uint16')
            elif col_max < 4294967295:
                df[col] = df[col].astype('uint32')
        else:
            if col_min >= -128 and col_max <= 127:
                df[col] = df[col].astype('int8')
            elif col_min >= -32768 and col_max <= 32767:
                df[col] = df[col].astype('int16')
            elif col_min >= -2147483648 and col_max <= 2147483647:
                df[col] = df[col].astype('int32')

    # Float kolonları optimize et
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')

    # String kolonları kategorik yap (eğer unique değer sayısı düşükse)
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # %50'den az unique değer varsa
            df[col] = df[col].astype('category')

    end_memory = df.memory_usage(deep=True).sum() / 1024**2

    print(f"📉 DataFrame memory optimized: {start_memory:.2f}MB → {end_memory:.2f}MB "
          f"({100 * (start_memory - end_memory) / start_memory:.1f}% reduction)")

    return df

def batch_processor(items: List, batch_size: int = 100, progress_callback: Optional[Callable] = None):
    """Büyük veri setlerini batch'ler halinde işle"""
    total_batches = (len(items) + batch_size - 1) // batch_size

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_num = i // batch_size + 1

        if progress_callback:
            progress_callback(batch_num, total_batches, len(batch))

        yield batch_num, batch

def memory_limit_check(max_memory_mb: int = 8192):
    """Bellek limiti kontrolü decorator'ü"""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_memory = optimizer.monitor_memory()

            if current_memory > max_memory_mb:
                print(f"⚠️ Memory limit exceeded: {current_memory:.1f}MB > {max_memory_mb}MB")
                optimizer.force_garbage_collection()

                current_memory = optimizer.monitor_memory()
                if current_memory > max_memory_mb:
                    raise MemoryError(f"Memory usage too high: {current_memory:.1f}MB")

            return func(*args, **kwargs)
        return wrapper
    return decorator

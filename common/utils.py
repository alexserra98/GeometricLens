import hashlib
import argparse
import json
import time
import functools

def _generate_hash(numpy_array):
    return hashlib.sha256(numpy_array.tobytes()).hexdigest()

def read_config(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error reading configuration file: {e}")
        return {}
    
def retry_on_failure(max_retries, delay=1):
    def decorator(func):
 
        def wrapper(*args, **kwargs):
            
            for _ in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    print(f"Error occurred: {e}. Retrying...")
                    time.sleep(delay)
                    delay *= 1.5
            raise Exception("Maximum retries exceeded. Function failed.")
        return wrapper
    return decorator

def retry_on_failure(max_retries,delay_time=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = delay_time
            for _ in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    print(f"Error occurred: {e}. Retrying...")
                    time.sleep(delay)
                    delay *= 1.5
                    
            raise Exception("Maximum retries exceeded. Function failed.")
        return wrapper
    return decorator

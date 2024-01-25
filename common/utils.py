import hashlib
import argparse
import json

def _generate_hash(numpy_array):
    return hashlib.sha256(numpy_array.tobytes()).hexdigest()

def read_config(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error reading configuration file: {e}")
        return {}
import yaml, os
from pathlib import Path

def read_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def create_directories(paths: list):
    for p in paths:
        os.makedirs(p, exist_ok=True)

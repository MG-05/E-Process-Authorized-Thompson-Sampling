"""Release inputs and integrity checks. All paths are relative to this checkout."""
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'

def array_hash(value, dtype):
    return hashlib.sha256(np.ascontiguousarray(value, dtype=dtype).tobytes()).hexdigest()

@lru_cache
def load_json(name):
    return json.loads((DATA/name).read_text())

def load_rows(suite):
    with gzip.open(DATA/f'{suite}.jsonl.gz', 'rt') as stream:
        return [json.loads(line) for line in stream]

def rng(descriptor):
    if descriptor['kind'] == 'integer':
        return np.random.default_rng(int(descriptor['value']))
    if descriptor['kind'] == 'numpy_seed_sequence':
        return np.random.default_rng(np.random.SeedSequence(
            descriptor['entropy'], spawn_key=tuple(descriptor['spawn_key']),
            pool_size=int(descriptor['pool_size'])))
    raise ValueError(f'Unsupported RNG descriptor: {descriptor}')

def verify_integrity():
    manifest = json.loads((ROOT/'provenance/release.json').read_text())
    for relative, expected in manifest['files'].items():
        path = ROOT/relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Release integrity failure: {relative}')
    return len(manifest['files'])

def output_directory(path):
    path = Path(path).resolve()
    # Never allow a rerun to overwrite release inputs or source.
    for protected in ('data', 'provenance', 'eats', 'tests', 'paper'):
        root = ROOT/protected
        if path == root or root in path.parents:
            raise ValueError(f'Output overlaps release inputs: {path}')
    if path == ROOT:
        raise ValueError('Choose a results subdirectory')
    path.mkdir(parents=True, exist_ok=True)
    return path

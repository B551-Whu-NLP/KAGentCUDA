"""Validate standalone cache entries against the exact original CUDA files."""

import hashlib
import json
from pathlib import Path


def validate_standalone_sources(entries, cuda_dir, manifest_path, cache_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    expected = manifest['cache_sha256']
    cache = Path(cache_path)
    if not cache.is_file() or hashlib.sha256(cache.read_bytes()).hexdigest() != expected:
        raise ValueError('Standalone cache differs from the source manifest')
    names = sorted({entry['filename'] for entry in entries})
    if len(entries) != manifest['kernel_count'] or len(names) != manifest['file_count']:
        raise ValueError('Standalone collection differs from the source manifest')
    root = Path(cuda_dir) / 'db_cuda_codes'
    validated = {}
    hashes = []
    for name in names:
        if Path(name).name != name or not name.endswith('.cu'):
            raise ValueError(f'Invalid standalone filename: {name}')
        source = root / name
        if not source.is_file():
            raise FileNotFoundError(f'Standalone CUDA source missing: {source}')
        raw = source.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        hashes.append(f'{name}\0{digest}\n')
        validated[name] = raw.decode('utf-8')
    source_digest = hashlib.sha256(''.join(hashes).encode()).hexdigest()
    if source_digest != manifest['sources_sha256']:
        raise ValueError('Standalone CUDA source collection differs from cache provenance')
    for entry in entries:
        snippet = entry['code'].strip().replace('\r\n', '\n')
        if not snippet or snippet not in validated[entry['filename']].replace('\r\n', '\n'):
            raise ValueError(f"Kernel snippet not found in original CUDA file: {entry['filename']}")
    return validated

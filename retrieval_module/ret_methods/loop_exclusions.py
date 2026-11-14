"""Explicit, source-bound review decisions for output-only C loops."""

import hashlib
import json
from pathlib import Path


def loop_digest(code):
    return hashlib.sha256(code.encode('utf-8')).hexdigest()


class LoopExclusions:
    def __init__(self, manifest_path):
        self.path = Path(manifest_path)
        document = json.loads(self.path.read_text(encoding='utf-8'))
        if document.get('version') != 1 or not isinstance(document.get('decisions'), list):
            raise ValueError('Loop review requires version 1 and a decisions array')
        self.decisions = {}
        for entry in document['decisions']:
            if (not isinstance(entry, dict) or entry.get('side') not in ('query', 'paired')
                    or not isinstance(entry.get('source'), str) or not entry['source']
                    or Path(entry['source']).is_absolute() or '..' in Path(entry['source']).parts
                    or not isinstance(entry.get('source_sha256'), str)
                    or len(entry['source_sha256']) != 64
                    or any(c not in '0123456789abcdef' for c in entry['source_sha256'])
                    or not isinstance(entry.get('loop_sha256'), str)
                    or len(entry['loop_sha256']) != 64
                    or any(c not in '0123456789abcdef' for c in entry['loop_sha256'])
                    or entry.get('decision') not in ('exclude', 'keep')
                    or not isinstance(entry.get('reason'), str) or not entry['reason'].strip()
                    or entry['reason'].startswith('review required:')):
                raise ValueError('Invalid loop review decision')
            key = (entry['side'], entry['source'], entry['source_sha256'], entry['loop_sha256'])
            if key in self.decisions:
                raise ValueError(f'Duplicate loop review decision: {key}')
            self.decisions[key] = entry['decision']

    def select(self, records, side, source_root, require_all=True):
        if side not in ('query', 'paired'):
            raise ValueError('Unknown review side')
        root = Path(source_root).resolve()
        retained, excluded, seen = [], [], set()
        source_hashes = {}
        for record in records:
            path = Path(record['filename']).resolve()
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError as exc:
                raise ValueError(f'Loop source outside review root: {path}') from exc
            if not path.is_file():
                raise FileNotFoundError(path)
            if path not in source_hashes:
                source_hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
            source_hash = source_hashes[path]
            code_hash = loop_digest(record['source_code'] if side == 'query' else record['code'])
            key = (side, relative, source_hash, code_hash)
            seen.add(key)
            decision = self.decisions.get(key)
            if decision is None:
                raise ValueError(f'Unreviewed {side} loop: {relative} {code_hash}')
            if decision == 'exclude':
                excluded.append(record)
            else:
                retained.append(record)
        reviewed_sources = {key[1] for key in self.decisions if key[0] == side}
        if any(Path(record['filename']).resolve().relative_to(root).as_posix() not in reviewed_sources
               for record in records):
            raise ValueError(f'Unreviewed {side} source in retrieval')
        missing = {key for key in self.decisions if key[0] == side} - seen
        if require_all and missing:
            raise ValueError(f'Review decisions not matched to {side} loops: {len(missing)}')
        return retained, excluded

    def validate_sources(self, side, source_root):
        root = Path(source_root).resolve()
        for decision_side, source, digest, _ in self.decisions:
            if decision_side != side:
                continue
            file = (root / source).resolve()
            if not file.is_relative_to(root) or not file.is_file():
                raise ValueError(f'Missing reviewed {side} source: {source}')
            if hashlib.sha256(file.read_bytes()).hexdigest() != digest:
                raise ValueError(f'Reviewed {side} source changed: {source}')


def reviewed_paired_loops(records, exclusions, source_root, obfuscate_code=False):
    from data_module.c_file_process import extract_loops
    from data_module.data_loader import obfuscate, C_KEYWORDS

    root = Path(source_root)
    source_lookup = {}
    for source in root.glob('type-*/*/c/*'):
        if source.is_file() and source.suffix in ('.c', '.cpp'):
            source_lookup.setdefault((source.parent.parent.name, source.name), []).append(source)
    reviewed = {}
    indexed = []
    for record in records:
        matches = source_lookup.get((record['project'], record['filename']), [])
        if len(matches) != 1:
            raise ValueError(f'Cannot uniquely resolve paired loop source: {record["project"]}/{record["filename"]}')
        source = matches[0]
        if source not in reviewed:
            raw = extract_loops(source)
            reviewed[source] = {}
            for loop in raw:
                cached = obfuscate(loop.source_code, C_KEYWORDS) if obfuscate_code else loop.source_code
                reviewed[source].setdefault(cached, []).append(loop.source_code)
        matches_raw = reviewed[source].get(record['code'], [])
        if not matches_raw:
            raise ValueError(f'Cached paired loop cannot be mapped to source: {record["id"]}')
        if len({loop_digest(code) for code in matches_raw}) > 1:
            raise ValueError(f'Ambiguous obfuscated loop identity in paired source: {source}')
        indexed.append({**record, 'filename': str(source), 'code': matches_raw.pop(0)})
    if any(codes for mapped in reviewed.values() for codes in mapped.values()):
        raise ValueError('Paired cache does not cover all extracted source loops')
    _, excluded = exclusions.select(indexed, 'paired', root)
    excluded_ids = {record['id'] for record in excluded}
    retained = [record for record in records if record['id'] not in excluded_ids]
    if not retained:
        raise ValueError('Output-only loop review left no paired loops')
    return retained, excluded

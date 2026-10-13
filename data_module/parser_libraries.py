"""Resolve user-supplied tree-sitter C/CUDA grammar libraries without redistributing binaries."""

import os
from pathlib import Path


_GRAMMARS = {
    'c': ('KAGENTCUDA_C_GRAMMAR_LIB', 'my-languages.so'),
    'cuda': ('KAGENTCUDA_CUDA_GRAMMAR_LIB', 'libtree-sitter-cuda.so'),
}


def grammar_library(language):
    env_name, filename = _GRAMMARS[language]
    override = os.environ.get(env_name)
    if override:
        supplied = Path(override).expanduser()
        if not supplied.is_absolute():
            raise ValueError(f'{env_name} must be an absolute path')
        path = supplied.resolve()
    else:
        path = Path(__file__).resolve().parents[1] / 'files' / filename
    if not path.is_file():
        raise FileNotFoundError(
            f'{language} tree-sitter grammar library missing: {path}. '
            f'Install a compatible, license-reviewed grammar for tree_sitter==0.20.4 '
            f'and set {env_name} to its absolute shared-library path. '
            'Third-party grammar libraries are not bundled with this repository.')
    return str(path)

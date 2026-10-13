"""Resolve complete paired examples from the unmodified source files."""

from pathlib import Path


def paired_programs(loader, base_path):
    base_path = Path(base_path)
    c_files = {}
    cuda_files = {}
    for entry in loader.c_codes:
        key = (entry.get('project'), entry.get('filename'))
        c_files[key] = entry
    for entry in loader.cuda_codes:
        key = (entry.get('project'), entry.get('filename'))
        cuda_files[key] = entry

    projects = {}
    for project, filename in c_files:
        projects.setdefault(project, [set(), set()])[0].add(filename)
    for project, filename in cuda_files:
        projects.setdefault(project, [set(), set()])[1].add(filename)

    result = {}
    for project, (cs, ks) in projects.items():
        if len(cs) != 1 or len(ks) != 1:
            continue
        c_name, k_name = next(iter(cs)), next(iter(ks))
        type_dirs = [p for p in base_path.glob('type-*') if (p / project / 'c' / c_name).is_file() and (p / project / 'cuda' / k_name).is_file()]
        if len(type_dirs) != 1:
            continue
        c_path = type_dirs[0] / project / 'c' / c_name
        k_path = type_dirs[0] / project / 'cuda' / k_name
        c_code = c_path.read_text(encoding='utf-8')
        cuda_code = k_path.read_text(encoding='utf-8')
        if c_code.strip() and cuda_code.strip() and '__global__' in cuda_code and 'main(' in cuda_code:
            result[(project, c_name)] = {'c_code': c_code, 'cuda_code': cuda_code, 'project': project,
                                         'c_path': str(c_path), 'cuda_path': str(k_path)}
    return result

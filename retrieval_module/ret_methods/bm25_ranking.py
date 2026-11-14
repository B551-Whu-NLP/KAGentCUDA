import math
import re
from collections import Counter
from pathlib import Path

from data_module.data_loader import obfuscate_ranking_code


class BM25:
    def __init__(self, corpus, obfuscate_code=True, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError('BM25 requires a nonempty corpus')
        self.obfuscate_code = obfuscate_code
        self.k1, self.b = k1, b
        self.frequencies = [Counter(self.tokenize(code)) for code in corpus]
        self.lengths = [sum(freq.values()) for freq in self.frequencies]
        self.avgdl = sum(self.lengths) / len(corpus)
        if not self.avgdl:
            raise ValueError('BM25 corpus has no tokens')
        df = Counter(token for freq in self.frequencies for token in freq)
        self.idf = {token: math.log(1 + (len(corpus) - count + 0.5) / (count + 0.5))
                    for token, count in df.items()}

    def tokenize(self, code):
        code = obfuscate_ranking_code(code) if self.obfuscate_code else code
        return [token.lower() for token in re.findall(
            r'[a-zA-Z_]\w*|\d+|<<|>>|<=|>=|==|!=|&&|\|\||[+\-*/%&|^~<>!=]', code)]

    def scores(self, query):
        tokens = Counter(self.tokenize(query))
        scores = []
        for freq, length in zip(self.frequencies, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * length / self.avgdl)
            scores.append(sum(count * self.idf.get(token, 0.0) *
                              freq.get(token, 0) * (self.k1 + 1) /
                              (freq.get(token, 0) + norm)
                              for token, count in tokens.items()))
        return scores

    def max_scores(self, queries):
        if not queries:
            raise ValueError('BM25 requires at least one query loop')
        return [max(values) for values in zip(*(self.scores(query) for query in queries))]


def validate_paired_snippets(records, source_root, side):
    root = Path(source_root).resolve()
    sources = {}
    for record in records:
        name = record['filename']
        if Path(name).name != name:
            raise ValueError(f'Invalid paired filename: {name}')
        project = record['project']
        if not isinstance(project, str) or Path(project).name != project or project in ('.', '..'):
            raise ValueError(f'Invalid paired project: {project}')
        key = (project, name)
        if key not in sources:
            matches = [path.resolve() for path in root.glob(f'type-*/{key[0]}/{side}/{name}')
                       if path.is_file()]
            if len(matches) != 1 or not matches[0].is_relative_to(root):
                raise ValueError(f'Cannot uniquely resolve paired source: {key}')
            sources[key] = matches[0].read_text(encoding='utf-8').replace('\r\n', '\n')
        snippet = record['code'].strip().replace('\r\n', '\n')
        if not snippet or snippet not in sources[key]:
            raise ValueError(f'Cached snippet not found in original paired source: {record["id"]}')
    return sources

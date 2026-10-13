"""Compute cumulative @k for independent samples or final trajectory outputs."""

import argparse
import json

CRITERIA = ('com', 'run', 'cor', 'fast')


def flags(candidate):
    if isinstance(candidate, dict) and 'layers' in candidate:
        layers = candidate['layers']
        return {name: all(layers.get(f'L{i}', {}).get('passed', False) for i in range(1, depth + 1))
                for name, depth in zip(CRITERIA, range(1, 5))}
    if not isinstance(candidate, dict) or any(name not in candidate for name in CRITERIA):
        raise ValueError('Candidate needs four boolean criteria or a pipeline layers record')
    return {name: bool(candidate[name]) for name in CRITERIA}


def rates(tasks, k, mode):
    if mode not in ('independent', 'trajectory') or k < 1:
        raise ValueError('Specify positive k and independent or trajectory mode')
    if not tasks:
        raise ValueError('At least one task is required')
    passed = {key: 0 for key in CRITERIA}
    for task in tasks:
        if mode == 'independent':
            attempts = task.get('candidates', [])
            if len(attempts) < k:
                raise ValueError('Independent tasks need at least k separate responses')
            outcomes = [flags(item) for item in attempts[:k]]
            aggregate = {name: any(outcome[name] for outcome in outcomes) for name in CRITERIA}
        else:
            checkpoints = task.get('final_at_response', [])
            if len(checkpoints) < k:
                raise ValueError('Trajectory needs a final output for each response budget')
            aggregate = flags(checkpoints[k - 1])
        for name in CRITERIA:
            passed[name] += aggregate[name]
    return {f'{name}@{k}': passed[name] / len(tasks) for name in CRITERIA}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='JSON containing a tasks array')
    parser.add_argument('--k', type=int, choices=(3, 5), required=True)
    parser.add_argument('--mode', choices=('independent', 'trajectory'), required=True)
    args = parser.parse_args()
    with open(args.input, encoding='utf-8') as stream:
        data = json.load(stream)
    print(json.dumps(rates(data['tasks'], args.k, args.mode), indent=2))

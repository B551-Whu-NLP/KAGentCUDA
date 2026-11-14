"""Validate the bundled synthetic adapter's reported sample shape, not external claims."""

import math
import statistics


def validate_timing_report(report, warmups=3, runs=3):
    if not isinstance(report, dict):
        raise ValueError('Timing adapter must return an evidence report, not two scalar times')
    required = ('protocol', 'cuda_scope', 'cpu_scope', 'cuda_clock', 'cpu_clock',
                'cuda_warmups_ms', 'cpu_warmups_ms', 'cuda_runs_ms', 'cpu_runs_ms',
                'synchronization', 'provenance')
    if any(field not in report for field in required):
        raise ValueError('Timing adapter omitted protocol or measurement evidence')
    if report['protocol'] != 'instrumented_end_to_end_v1':
        raise ValueError('Unknown timing protocol')
    if (report['cuda_scope'] != 'H2D+kernel+D2H' or report['cpu_scope'] != 'equivalent_complete_computation'
            or report['cuda_clock'] != 'cudaEvent' or report['cpu_clock'] != 'monotonic_host'
            or report['synchronization'] != 'event_after_last_D2H_and_stream_synchronized'
            or not isinstance(report['provenance'], str) or not report['provenance'].strip()):
        raise ValueError('Timing boundaries or provenance not established')
    for side in ('cuda', 'cpu'):
        for stage, count in (('warmups', warmups), ('runs', runs)):
            values = report[f'{side}_{stage}_ms']
            if (not isinstance(values, list) or len(values) != count
                    or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                           for value in values)):
                raise ValueError(f'{side} {stage} must have {count} positive finite samples')
    return statistics.median(report['cuda_runs_ms']), statistics.median(report['cpu_runs_ms'])

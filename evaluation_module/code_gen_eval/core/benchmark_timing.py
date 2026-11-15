import re
import subprocess
from pathlib import Path


CUDA_WRAPPER = r'''
#include <cstdio>
#include <cuda_runtime.h>

int kagentcuda_timed_main(int argc, char **argv);

int main(int argc, char **argv) {
    cudaEvent_t start, end;
    if (cudaEventCreate(&start) != cudaSuccess) return 90;
    if (cudaEventCreate(&end) != cudaSuccess) return 90;
    if (cudaEventRecord(start, 0) != cudaSuccess) return 90;
    int status = kagentcuda_timed_main(argc, argv);
    if (cudaEventRecord(end, 0) != cudaSuccess) return 90;
    if (cudaEventSynchronize(end) != cudaSuccess) return 90;
    float milliseconds = 0.0f;
    if (cudaEventElapsedTime(&milliseconds, start, end) != cudaSuccess) return 90;
    std::fprintf(stderr, "KAGENTCUDA_TIMING_MS=%.9g\n", milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return status;
}
'''


CPU_WRAPPER = r'''
#include <stdio.h>
#include <time.h>

int kagentcuda_timed_main(int argc, char **argv);

static double elapsed_ms(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) * 1000.0 +
           (end.tv_nsec - start.tv_nsec) / 1000000.0;
}

int main(int argc, char **argv) {
    struct timespec start, end;
    if (clock_gettime(CLOCK_MONOTONIC, &start) != 0) return 90;
    int status = kagentcuda_timed_main(argc, argv);
    if (clock_gettime(CLOCK_MONOTONIC, &end) != 0) return 90;
    fprintf(stderr, "KAGENTCUDA_TIMING_MS=%.9g\n", elapsed_ms(start, end));
    return status;
}
'''


class BenchmarkTiming:
    def __init__(self, cuda_source, cpu_source, workdir, timeout=300, include_path=None):
        self.cuda_source = Path(cuda_source)
        self.cpu_source = Path(cpu_source)
        self.workdir = Path(workdir)
        self.timeout = timeout
        self.include_path = Path(include_path) if include_path else None

    def _compile(self, source, wrapper_text, side):
        if side == 'cuda':
            compiler = ['nvcc', '-O3', '-arch=sm_89']
            wrapper_suffix = '.cu'
        else:
            compiler = ['gcc', '-O3', '-D_POSIX_C_SOURCE=200809L']
            wrapper_suffix = '.c'
        include = ['-I', str(self.include_path.resolve())] if self.include_path else []
        object_path = self.workdir / f'{side}_original.o'
        wrapper_path = self.workdir / f'{side}_wrapper{wrapper_suffix}'
        binary_path = self.workdir / f'{side}_timed'
        wrapper_path.write_text(wrapper_text, encoding='utf-8')
        commands = [
            compiler + ['-Dmain=kagentcuda_timed_main', *include,
                        '-c', str(source.resolve()), '-o', str(object_path.resolve())],
            compiler + [str(wrapper_path.resolve()), str(object_path.resolve()),
                        '-o', str(binary_path.resolve()), '-lm'],
        ]
        for command in commands:
            proc = subprocess.run(command, cwd=self.workdir, capture_output=True,
                                  text=True, timeout=self.timeout)
            if proc.returncode != 0:
                raise RuntimeError(f'{side} timing compilation failed: {proc.stdout}\n{proc.stderr}')
        return binary_path

    def _measure(self, binary, warmups, runs):
        values = []
        for _ in range(warmups + runs):
            proc = subprocess.run([str(binary.resolve())], cwd=self.workdir,
                                  capture_output=True, text=True, timeout=self.timeout)
            matches = re.findall(r'^KAGENTCUDA_TIMING_MS=([0-9.eE+-]+)$',
                                 proc.stderr, re.MULTILINE)
            if proc.returncode != 0 or len(matches) != 1:
                raise RuntimeError(f'{binary.name} timing failed: {proc.stdout}\n{proc.stderr}')
            values.append(float(matches[0]))
        return values

    def __call__(self, cuda_binary, cpu_binary, warmups, runs):
        if not self.cuda_source.is_file() or not self.cpu_source.is_file():
            raise ValueError('Benchmark timing requires the original C and CUDA sources')
        self.workdir.mkdir(parents=True, exist_ok=True)
        cuda_exe = self._compile(self.cuda_source, CUDA_WRAPPER, 'cuda')
        cpu_exe = self._compile(self.cpu_source, CPU_WRAPPER, 'cpu')
        cuda_samples = self._measure(cuda_exe, warmups, runs)
        cpu_samples = self._measure(cpu_exe, warmups, runs)
        return {
            'protocol': 'instrumented_end_to_end_v1',
            'cuda_scope': 'H2D+kernel+D2H',
            'cpu_scope': 'equivalent_complete_computation',
            'cuda_clock': 'cudaEvent',
            'cpu_clock': 'monotonic_host',
            'synchronization': 'event_after_last_D2H_and_stream_synchronized',
            'cuda_warmups_ms': cuda_samples[:warmups],
            'cpu_warmups_ms': cpu_samples[:warmups],
            'cuda_runs_ms': cuda_samples[warmups:],
            'cpu_runs_ms': cpu_samples[warmups:],
            'provenance': 'Compiled wrapper around the original C and CUDA main functions',
        }

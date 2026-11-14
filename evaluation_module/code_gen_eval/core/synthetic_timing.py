"""Opt-in instrumented timing of the original synthetic vector-add illustration only."""

import hashlib
import re
import subprocess
from pathlib import Path


CUDA_SHA256 = 'b8ff3cf589014d138fd6f9cc07d4d340ba7b4c696cbe373a22a38cf1d5419a7b'
CPU_SHA256 = 'a9c12d981c5e767bb2e53a2ac8678b136adbeebd9f17cc688a208b7712bfa16d'
EXPECTED_OUTPUT = '6\n8\n10\n12\n'

CUDA_PROGRAM = r'''
#include <cstdio>
#include <cuda_runtime.h>
#define CHECK(call) do { cudaError_t status = (call); if (status != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status)); return 2; } } while (0)
__global__ void vector_add(const int *a, const int *b, int *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
int main() {
    const int n = 4;
    const size_t bytes = n * sizeof(int);
    int *a = nullptr, *b = nullptr, *out = nullptr;
    int *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
    CHECK(cudaMallocHost((void **)&a, bytes));
    CHECK(cudaMallocHost((void **)&b, bytes));
    CHECK(cudaMallocHost((void **)&out, bytes));
    for (int i = 0; i < n; ++i) { a[i] = i + 1; b[i] = i + 5; }
    CHECK(cudaMalloc((void **)&d_a, bytes));
    CHECK(cudaMalloc((void **)&d_b, bytes));
    CHECK(cudaMalloc((void **)&d_out, bytes));
    cudaEvent_t start, end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(start, 0));
    CHECK(cudaMemcpyAsync(d_a, a, bytes, cudaMemcpyHostToDevice, 0));
    CHECK(cudaMemcpyAsync(d_b, b, bytes, cudaMemcpyHostToDevice, 0));
    vector_add<<<1, n, 0, 0>>>(d_a, d_b, d_out, n);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpyAsync(out, d_out, bytes, cudaMemcpyDeviceToHost, 0));
    CHECK(cudaEventRecord(end, 0));
    CHECK(cudaEventSynchronize(end));
    float milliseconds = 0.0f;
    CHECK(cudaEventElapsedTime(&milliseconds, start, end));
    fprintf(stderr, "TIMING_MS=%.9g\n", milliseconds);
    for (int i = 0; i < n; ++i) printf("%d\n", out[i]);
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(end));
    CHECK(cudaFree(d_a)); CHECK(cudaFree(d_b)); CHECK(cudaFree(d_out));
    CHECK(cudaFreeHost(a)); CHECK(cudaFreeHost(b)); CHECK(cudaFreeHost(out));
    return 0;
}
'''

CPU_PROGRAM = r'''
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <time.h>
int main(void) {
    volatile int a[4] = {1, 2, 3, 4}, b[4] = {5, 6, 7, 8};
    volatile int out[4];
    struct timespec start, end;
    if (clock_gettime(CLOCK_MONOTONIC, &start)) return 2;
    for (int i = 0; i < 4; ++i) out[i] = a[i] + b[i];
    if (clock_gettime(CLOCK_MONOTONIC, &end)) return 2;
    double milliseconds = (end.tv_sec - start.tv_sec) * 1000.0 +
                          (end.tv_nsec - start.tv_nsec) / 1000000.0;
    fprintf(stderr, "TIMING_MS=%.9g\n", milliseconds);
    for (int i = 0; i < 4; ++i) printf("%d\n", out[i]);
    return 0;
}
'''


class SyntheticVectorAddTiming:
    def __init__(self, cuda_source, cpu_source, workdir, timeout=300):
        self.cuda_source = Path(cuda_source)
        self.cpu_source = Path(cpu_source)
        self.workdir = Path(workdir)
        self.timeout = timeout

    def validate_sources(self):
        if (not self.cuda_source.is_file() or not self.cpu_source.is_file() or
                hashlib.sha256(self.cuda_source.read_bytes()).hexdigest() != CUDA_SHA256 or
                hashlib.sha256(self.cpu_source.read_bytes()).hexdigest() != CPU_SHA256):
            raise ValueError('Only the exact original synthetic C/CUDA vector-add pair is supported')

    def __call__(self, cuda_binary, cpu_binary, warmups, runs):
        if warmups != 3 or runs != 3:
            raise ValueError('Synthetic timing requires exactly three warmups and three samples')
        self.validate_sources()
        if not Path(cuda_binary).is_file() or not Path(cpu_binary).is_file():
            raise ValueError('L4 compiled binaries must exist before instrumented timing')
        self.workdir.mkdir(parents=True, exist_ok=True)
        cuda_src = self.workdir / 'synthetic_vector_add_timed.cu'
        cpu_src = self.workdir / 'synthetic_vector_add_timed.c'
        cuda_src.write_text(CUDA_PROGRAM, encoding='utf-8')
        cpu_src.write_text(CPU_PROGRAM, encoding='utf-8')
        cuda_exe = self.workdir / 'synthetic_vector_add_timed_cuda'
        cpu_exe = self.workdir / 'synthetic_vector_add_timed_cpu'
        for command in (['nvcc', '-O3', '-arch=sm_89', str(cuda_src), '-o', str(cuda_exe)],
                        ['gcc', '-O3', str(cpu_src), '-o', str(cpu_exe)]):
            result = subprocess.run(command, cwd=self.workdir, capture_output=True, text=True,
                                    timeout=self.timeout)
            if result.returncode:
                raise RuntimeError(f'Synthetic timing compilation failed: {result.stderr}')
        samples = {}
        for side, binary, original in (('cuda', cuda_exe, Path(cuda_binary)),
                                       ('cpu', cpu_exe, Path(cpu_binary))):
            baseline = subprocess.run([str(original.resolve())], cwd=self.workdir,
                                      capture_output=True, text=True, timeout=self.timeout)
            if baseline.returncode or baseline.stdout != EXPECTED_OUTPUT:
                raise RuntimeError(f'Original synthetic {side} binary does not match instrumented output')
            values = []
            for _ in range(warmups + runs):
                result = subprocess.run([str(binary)], cwd=self.workdir, capture_output=True,
                                        text=True, timeout=self.timeout)
                matches = re.findall(r'^TIMING_MS=([0-9.eE+-]+)$', result.stderr, re.MULTILINE)
                if result.returncode or result.stdout != EXPECTED_OUTPUT or len(matches) != 1:
                    raise RuntimeError(f'Synthetic {side} timing or output check failed')
                values.append(float(matches[0]))
            samples[side] = values
        return {
            'protocol': 'instrumented_end_to_end_v1',
            'cuda_scope': 'H2D+kernel+D2H', 'cpu_scope': 'equivalent_complete_computation',
            'cuda_clock': 'cudaEvent', 'cpu_clock': 'monotonic_host',
            'synchronization': 'event_after_last_D2H_and_stream_synchronized',
            'cuda_warmups_ms': samples['cuda'][:warmups], 'cuda_runs_ms': samples['cuda'][warmups:],
            'cpu_warmups_ms': samples['cpu'][:warmups], 'cpu_runs_ms': samples['cpu'][warmups:],
            'provenance': 'Exact-hash synthetic vector-add pair; audited instrumented build; not a validated benchmark',
        }

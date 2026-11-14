#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void add_sources_d(const float* const model, float* wfp, const float* const source_amplitude,
    const int* const sources_z, const int* const sources_x,
    const int nz, const int nx, const int nt, const int ns, const int it) {

    for (int b = 0; b < ns; b++) {
        const float sa = source_amplitude[b * nt + it];
        const int* sz = sources_z + b * nx;
        const int* sx = sources_x + b * nx;

        for (int x = 0; x < nx; x++) {
            int i = sz[x] * nx + sx[x];
            wfp[i] += sa * model[i];
        }
    }
}

int main(int argc, char* argv[]) {
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }
    int nz = 1000;
    int nx = 1000;
    int nt = 1000;
    int ns = 1000;
    int it = 10;

    float* h_model          = (float*)malloc(nz * nx * sizeof(float));
    float* h_wfp            = (float*)malloc(nz * nx * sizeof(float));
    float* h_source_amplitude = (float*)malloc(ns * nt * sizeof(float));
    int*   h_sources_z      = (int*)malloc(ns * nx * sizeof(int));
    int*   h_sources_x      = (int*)malloc(ns * nx * sizeof(int));

    srand(42);
    for (int i = 0; i < nz * nx; ++i) {
        h_model[i] = (float)(rand() % 100) / 100.0f;
        h_wfp[i]   = 0.0f;
    }
    for (int i = 0; i < ns * nt; ++i)
        h_source_amplitude[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < ns * nx; ++i) {
        h_sources_z[i] = rand() % nz;
        h_sources_x[i] = rand() % nx;
    }

    clock_t start = clock();
    add_sources_d(h_model, h_wfp, h_source_amplitude,
                  h_sources_z, h_sources_x, nz, nx, nt, ns, it);
    clock_t end = clock();

    printf("Time: %.2f(ms)\n", ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0);

    if (verbose) {
        FILE* file = fopen("c_136_output.txt", "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                for (int j = 0; j < 1000; ++j)
                    fprintf(file, "%.6f ", h_wfp[i * nx + j]);
                fprintf(file, "\n");
            }
            fclose(file);
        }
    }

    free(h_model);
    free(h_wfp);
    free(h_source_amplitude);
    free(h_sources_z);
    free(h_sources_x);
    return 0;
}
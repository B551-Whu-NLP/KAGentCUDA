#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void laplace_filter(float* Img, float* laplace, float _dz, float _dx, int npml, int nnz, int nnx) {
    clock_t start = clock();

    for (int i1 = npml; i1 < nnz - npml; i1++) {
        for (int i2 = npml; i2 < nnx - npml; i2++) {
            int id = i1 + i2 * nnz;
            float diff1 = 0.0f;
            float diff2 = 0.0f;
            diff1 = Img[id + 1] - 2.0 * Img[id] + Img[id - 1];
            diff2 = Img[id + nnz] - 2.0 * Img[id] + Img[id - nnz];
            laplace[id] = _dz * _dz * diff1 + _dx * _dx * diff2;
        }
    }

    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time: %.2f\n", time_taken);
}

int main(int argc, char** argv) {
    const int nnz = 10000;
    const int nnx = 10000;
    const int npml = 2;

    float* Img_host = (float*)malloc(nnz * nnx * sizeof(float));
    float* laplace_host = (float*)malloc(nnz * nnx * sizeof(float));

    for (int i = 0; i < nnz * nnx; i++) {
        Img_host[i] = (float)(i % 256);
    }
    for (int i = 0; i < nnz * nnx; i++) {
        laplace_host[i] = (float)0;
    }

    laplace_filter(Img_host, laplace_host, 0.1, 0.2, npml, nnz, nnx);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        FILE* fp = fopen("c_332_output.txt", "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                for (int j = 0; j < 1000; j++) {
                    fprintf(fp, "%.6f ", laplace_host[i * nnx + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(Img_host);
    free(laplace_host);
    return 0;
}
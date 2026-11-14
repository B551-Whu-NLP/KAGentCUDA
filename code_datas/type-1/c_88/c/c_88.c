#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void devidecountInner(long Xsize, long Ysize, long Zsize, double* p, double* pn, int* pcountinner, int verbose) {
    long totalSize = Xsize * Ysize * Zsize;
    
    clock_t start = clock();

    for (int iter = 0; iter < 1000; iter++) {
        for (long tid = 0; tid < totalSize; tid++) {
            if (pcountinner[tid] > 1) {
                p[tid] = pn[tid] / pcountinner[tid];
                pn[tid] = 0;
            }
        }
    }

    clock_t end = clock();
    double timeElapsed = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", timeElapsed);

    if (verbose) {
        char outputFileName[256];
        sprintf(outputFileName, "c_88_output.txt");
        FILE* file = fopen(outputFileName, "w");
        if (file != NULL) {
            for (long i = 0; i < 1000; ++i) {
                fprintf(file, "p[%ld] = %f, pn[%ld] = %f, pcountinner[%ld] = %d\n", i, p[i], i, pn[i], i, pcountinner[i]);
            }
            fclose(file);
        }
    }
}

int main(int argc, char* argv[]) {
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    long Xsize = 200;
    long Ysize = 200;
    long Zsize = 200;
    long totalSize = Xsize * Ysize * Zsize;
    double* h_p = (double*)malloc(totalSize * sizeof(double));
    double* h_pn = (double*)malloc(totalSize * sizeof(double));
    int* h_pcountinner = (int*)malloc(totalSize * sizeof(int));

    srand(42);

    for (long i = 0; i < totalSize; ++i) {
        h_p[i] = 0.0;
        h_pn[i] = (double)(rand() % 100);
        h_pcountinner[i] = rand() % 10 + 1;
    }

    devidecountInner(Xsize, Ysize, Zsize, h_p, h_pn, h_pcountinner, verbose);

    free(h_p);
    free(h_pn);
    free(h_pcountinner);

    return 0;
}
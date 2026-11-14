#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void devidecount(long Xsize, long Ysize, long Zsize, double *pint, int *pcount) {
    long n = Xsize * Ysize * 2 + (Ysize - 2) * Zsize * 2 + (Xsize - 2) * (Zsize - 2) * 2;
    
    for (long tid = 0; tid < n * n; tid++) {
        if (pcount[tid] > 1) {
            pint[tid] /= pcount[tid];
        }
    }
}

int main(int argc, char *argv[]) {
    long Xsize = 10, Ysize = 10, Zsize = 10;
    long n = Xsize * Ysize * 2 + (Ysize - 2) * Zsize * 2 + (Xsize - 2) * (Zsize - 2) * 2;
    size_t size = n * n * sizeof(double);
    size_t countSize = n * n * sizeof(int);
    
    double* h_pint = (double*)malloc(size);
    int* h_pcount = (int*)malloc(countSize);
    
    if (!h_pint || !h_pcount) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    clock_t start = clock();

    for (long i = 0; i < n * n; i++) {
        h_pint[i] = 1.0;
        h_pcount[i] = 2;
    }

    devidecount(Xsize, Ysize, Zsize, h_pint, h_pcount);

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_92_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (long i = 0; i < 100; i++) {
                for (long j = 0; j < 100; j++) {
                    fprintf(fp, "%.6f ", h_pint[i * n + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

 
    printf("Time: %.2f(ms)\n", time_taken);
    free(h_pint);
    free(h_pcount);

    return 0;
}
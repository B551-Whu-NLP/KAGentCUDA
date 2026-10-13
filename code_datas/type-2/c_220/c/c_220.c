#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void record(float* p, float* seis_kt, int* Gxz, int ng) {
    for (int id = 0; id < ng; id++) {
        seis_kt[id] = p[Gxz[id]];
    }
}

int main(int argc, char *argv[]) {
    const int array_size = 1000000;
    float* p_host = (float*)malloc(array_size * sizeof(float));
    float* seis_kt_host = (float*)malloc(array_size * sizeof(float));
    int* Gxz_host = (int*)malloc(array_size * sizeof(int));

    for (int i = 0; i < array_size; ++i) {
        p_host[i] = i + 1.0;
        seis_kt_host[i] = 0.0;
        Gxz_host[i] = i % array_size;
    }

    clock_t start = clock();
    record(p_host, seis_kt_host, Gxz_host, array_size);
    clock_t end = clock();

    double time_taken = ((double)(end - start) * 1000) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_220_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%.2f ", seis_kt_host[i]);
            }
            fclose(fp);
        }
    }

    free(p_host);
    free(seis_kt_host);
    free(Gxz_host);

    return 0;
}
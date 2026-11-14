#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void circularity(const int compCount, const int* areaRes, const float* perimeterRes, float* circ) {
    for (int tid = 0; tid < compCount; tid++) {
        circ[tid] = (4.0 * 3.14159265359 * (float)areaRes[tid]) / (perimeterRes[tid] * perimeterRes[tid]);
    }
}

int main(int argc, char *argv[]) {
    int compCount = 1000000;
    int* h_areaRes = (int*)malloc(compCount * sizeof(int));
    float* h_perimeterRes = (float*)malloc(compCount * sizeof(float));
    float* h_circ = (float*)malloc(compCount * sizeof(float));

    for (int i = 0; i < compCount; i++) {
        h_areaRes[i] = i + 1;
        h_perimeterRes[i] = (float)(i + 1) * 2.0;
    }

    clock_t start = clock();

    circularity(compCount, h_areaRes, h_perimeterRes, h_circ);

    clock_t end = clock();
    double time_spent = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2fms\n", time_spent * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_91_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "Circularity for component %d: %f\n", i, h_circ[i]);
            }
            fclose(fp);
        }
    }

    free(h_areaRes);
    free(h_perimeterRes);
    free(h_circ);

    return 0;
}
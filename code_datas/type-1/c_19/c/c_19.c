#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void mul_Scalar_matrix(float* a, float value, float* c, int N) {
    for (int idx = 0; idx < N; idx++) {
        c[idx] = a[idx] * value;
    }
}

int main(int argc, char *argv[]) {
    int arraySize = 10000000; // 增大数组大小
    float value = 2.0;
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    float* h_a = (float*)malloc(arraySize * sizeof(float));
    float* h_c = (float*)malloc(arraySize * sizeof(float));

    for (int i = 0; i < arraySize; ++i) {
        h_a[i] = (float)(i);
    }

    clock_t start = clock();

    mul_Scalar_matrix(h_a, value, h_c, arraySize);

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_19_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%.2f ", h_c[i]);
            }
            fclose(fp);
        }
    }

    free(h_a);
    free(h_c);
    return 0;
}
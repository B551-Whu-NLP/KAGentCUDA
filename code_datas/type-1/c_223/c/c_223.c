#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void vectorAdd(const float* A, const float* B, float* C, int numElements) {
    for (int i = 0; i < numElements; i++) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char* argv[]) {
    const int array_size = 10000000; // 增大数组大小
    float* A_host = (float*)malloc(array_size * sizeof(float));
    float* B_host = (float*)malloc(array_size * sizeof(float));
    float* C_host = (float*)malloc(array_size * sizeof(float));

    for (int i = 0; i < array_size; ++i) {
        A_host[i] = i + 1.0;
        B_host[i] = (i + 1.0) * 2.0;
    }

    clock_t start = clock();

    vectorAdd(A_host, B_host, C_host, array_size);

    clock_t end = clock();

    double time_spent = ((double)(end - start)) / CLOCKS_PER_SEC;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        FILE* output_file = fopen("c_223_output.txt", "w");
        if (output_file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(output_file, "%.2f ", C_host[i]);
            }
            fclose(output_file);
        }
    }

    free(A_host);
    free(B_host);
    free(C_host);

    printf("Time: %.2f(ms)\n", time_spent * 1000);

    return 0;
}
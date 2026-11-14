#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void sumArrays(float* A, float* B, float* C, const int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char* argv[]) {
    const int array_size = 10000000; // 增大数组大小
    float* A_host = (float*)malloc(array_size * sizeof(float));
    float* B_host = (float*)malloc(array_size * sizeof(float));
    float* C_result = (float*)malloc(array_size * sizeof(float));

    for (int i = 0; i < array_size; ++i) {
        A_host[i] = i + 1.0;
        B_host[i] = (i + 1.0) * 2.0;
    }

    clock_t start = clock();
    sumArrays(A_host, B_host, C_result, array_size);
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    printf("Time: %.2fms\n", time_taken * 1000);
    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_214_output.txt");
        FILE* file = fopen(output_file, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "%.2f ", C_result[i]);
            }
            fclose(file);
        }
    }

    free(A_host);
    free(B_host);
    free(C_result);

    return 0;
}
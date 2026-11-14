#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void addVectorsInto(float* result, float* a, float* b, int N) {
    for (int i = 0; i < N; i++) {
        result[i] = a[i] + b[i];
    }
}

int main(int argc, char* argv[]) {
    int N = 10000000; // 增加数组大小以延长执行时间
    float* h_result = (float*)malloc(N * sizeof(float));
    float* h_a = (float*)malloc(N * sizeof(float));
    float* h_b = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        h_a[i] = i + 1;
        h_b[i] = 2 * (i + 1);
    }

    clock_t start = clock();

    addVectorsInto(h_result, h_a, h_b, N);

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_240_output.txt");
        FILE* file = fopen(output_file, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "%f ", h_result[i]);
            }
            fclose(file);
        }
    } 

    free(h_result);
    free(h_a);
    free(h_b);

    printf("Time: %.2f(ms)\n", time_taken * 1000);

    return 0;
}
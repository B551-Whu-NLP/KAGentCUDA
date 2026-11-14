#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void sigmoid_kernel(float* input, float* output, int n) {
    for (int tid = 0; tid < n; tid++) {
        output[tid] = 1.0 / (1.0 + expf(-input[tid]));
    }
}

int main(int argc, char** argv) {
    int data_size = 1000000;
    float* input_host = (float*)malloc(data_size * sizeof(float));
    float* output_host = (float*)malloc(data_size * sizeof(float));
    for (int i = 0; i < data_size; ++i) {
        input_host[i] = i % 1000 + 1.0;
    }

    clock_t start = clock();

    sigmoid_kernel(input_host, output_host, data_size);

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_195_output.txt");
        FILE* f = fopen(output_file, "w");
        if (f != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(f, "%.6f ", output_host[i]);
            }
            fprintf(f, "\n");
            fclose(f);
        }
    }

    free(input_host);
    free(output_host);

    return 0;
}
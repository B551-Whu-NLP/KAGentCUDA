#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void boxesScale(const float* input, float* output, int dims, float scale0, float scale1, float scale2, float scale3) {
    for (int tid = 0; tid < dims; tid++) {
        output[tid * 4] = input[tid * 4] / scale0;
        output[tid * 4 + 1] = input[tid * 4 + 1] / scale1;
        output[tid * 4 + 2] = input[tid * 4 + 2] / scale2;
        output[tid * 4 + 3] = input[tid * 4 + 3] / scale3;
    }
}

int main(int argc, char *argv[]) {
    int dims = 1000000; 
    float scale0 = 2.0; 
    float scale1 = 1.5; 
    float scale2 = 3.0; 
    float scale3 = 0.5; 

    float *input = (float *)malloc(dims * 4 * sizeof(float));
    float *output = (float *)malloc(dims * 4 * sizeof(float));

    if (input == NULL || output == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    for (int i = 0; i < dims * 4; i++) {
        input[i] = i + 1.0;
    }

    clock_t start = clock();

    boxesScale(input, output, dims, scale0, scale1, scale2, scale3);

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", time_taken);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_319_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%.6f ", output[i]);
            }
            fprintf(fp, "\n");
            fclose(fp);
        }
    }

    free(input);
    free(output);

    return 0;
}
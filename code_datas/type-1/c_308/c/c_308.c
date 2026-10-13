#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void expandBoxes(const float* input, float* output, int dims, int clsNum) {
    for (int tid = 0; tid < dims; tid++) {
        int k = tid / clsNum;
        output[tid * 4 + 0] = input[k * 4 + 0];
        output[tid * 4 + 1] = input[k * 4 + 1];
        output[tid * 4 + 2] = input[k * 4 + 2];
        output[tid * 4 + 3] = input[k * 4 + 3];
    }
}

int main(int argc, char *argv[]) {
    int dims = 1200000;
    int clsNum = 400;
    float *input = (float *)malloc(clsNum * 4 * sizeof(float));
    float *output = (float *)malloc(dims * 4 * sizeof(float));
    
    for (int i = 0; i < clsNum * 4; i++) {
        input[i] = i + 1.0; 
    }

    clock_t start = clock();
    expandBoxes(input, output, dims, clsNum);
    clock_t end = clock();
    
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_308_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                for (int j = 0; j < 4; j++) {
                    fprintf(fp, "%.2f ", output[i * 4 + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(input);
    free(output);
    printf("Time: %.2f(ms)\n", time_taken * 1000);
    return 0;
}
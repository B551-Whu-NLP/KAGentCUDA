#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void resizedClsScore(const float* score, const float* score_factors, float* output, int dims) {
    for (int tid = 0; tid < dims; tid++) {
        if (score[tid] == (-1)) {
            output[tid] = -1;
        } else {
            output[tid] = score[tid] * score_factors[tid];
        }
    }
}

int main(int argc, char* argv[]) {
    int dims = 1024 * 1000; 
    float* h_score = (float*)malloc(dims * sizeof(float));
    float* h_score_factors = (float*)malloc(dims * sizeof(float));
    float* h_output = (float*)malloc(dims * sizeof(float));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < dims; ++i) {
        h_score[i] = (float)(i % 10);
        h_score_factors[i] = (float)((i % 2) + 1);
    }

    clock_t start = clock();

    for (int iter = 0; iter < 10000; iter++) {
        resizedClsScore(h_score, h_score_factors, h_output, dims);
    }

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    if (verbose) {
        char output_filename[256];
        sprintf(output_filename, "%s_output.txt", "c_78");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "output[%d]: %f\n", i, h_output[i]);
            }
            fclose(file);
        }
    }

    printf("Time: %.2f(ms)\n", time_taken * 1000);
    

    free(h_score);
    free(h_score_factors);
    free(h_output);
    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void set_valid_mask(const float *score, float score_thr, int *valid_mask, int dims) {
    for (int tid = 0; tid < dims; tid++) {
        if (score[tid] > score_thr) {
            valid_mask[tid] = 1;
        } else {
            valid_mask[tid] = 0;
        }
    }
}

int main(int argc, char *argv[]) {
    const int dims = 2048000; // 增大数组大小
    float* h_score = (float*)malloc(dims * sizeof(float));
    int* h_valid_mask = (int*)malloc(dims * sizeof(int));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < dims; ++i) {
        h_score[i] = (float)(i);
    }

    clock_t start = clock();

    // 多次调用函数以增加总执行时间
    for (int iter = 0; iter < 1000; iter++) {
        set_valid_mask(h_score, 500.0f, h_valid_mask, dims);
    }

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    if (verbose) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "c_67_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "h_valid_mask[%d]: %d\n", i, h_valid_mask[i]);
            }
            fclose(fp);
        }
    }

    printf("Time: %.2f(ms)\n", time_taken * 1000);
    

    free(h_score);
    free(h_valid_mask);

    return 0;
}
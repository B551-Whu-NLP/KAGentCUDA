#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void l1(int n, float* pred, float* truth, float* delta, float* error) {
    for (int i = 0; i < n; ++i) {
        float diff = truth[i] - pred[i];
        error[i] = fabs(diff);
        delta[i] = diff > 0 ? 1 : -1;
    }
}

int main(int argc, char* argv[]) {
    int n = 16777216; // 进一步增大数组大小以显著增加计算量
    float* h_pred = (float*)malloc(n * sizeof(float));
    float* h_truth = (float*)malloc(n * sizeof(float));
    float* h_delta = (float*)malloc(n * sizeof(float));
    float* h_error = (float*)malloc(n * sizeof(float));
    
    srand(42);
    
    for (int i = 0; i < n; ++i) {
        h_pred[i] = (float)(rand() % 100);
        h_truth[i] = (float)(rand() % 100);
    }

    clock_t start = clock();

    l1(n, h_pred, h_truth, h_delta, h_error);

    clock_t end = clock();

    double time_spent = ((double)(end - start)) / CLOCKS_PER_SEC;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_79_output.txt");
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "delta[%d]: %f, error[%d]: %f\n", i, h_delta[i], i, h_error[i]);
            }
            fclose(fp);
        }
    } 

    free(h_pred);
    free(h_truth);
    free(h_delta);
    free(h_error);

    printf("Time: %.2f(ms)\n", time_spent * 1000);

    return 0;
}
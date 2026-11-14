#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void binarize(float* x, int n, float* binary) {
    for (int i = 0; i < n; i++) {
        binary[i] = (x[i] > 0) ? 1.0f : -1.0f;
    }
}

int main(int argc, char* argv[]) {
    int n = 100000000; // 增加100倍
    float* h_x;
    float* h_binary;
    h_x = (float*)malloc(n * sizeof(float));
    h_binary = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n; ++i) {
        h_x[i] = i - 50;
    }

    clock_t start = clock();

    binarize(h_x, n, h_binary);

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_230_output.txt");
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%f ", h_binary[i]);
            }
            fclose(fp);
        }
    }

    free(h_x);
    free(h_binary);

    return 0;
}
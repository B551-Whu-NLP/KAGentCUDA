#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void incrementArray(float* a, int N) {
    for (int i = 0; i < N; i++) {
        a[i] = a[i] + 1.0f;
    }
}

int main(int argc, char *argv[]) {
    const int data_size = 100000000; // 增加100倍
    float* a_host = (float*)malloc(data_size * sizeof(float));
    for (int i = 0; i < data_size; ++i) {
        a_host[i] = i + 1.0;
    }
    clock_t start = clock();
    incrementArray(a_host, data_size);
    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }
    if (verbose) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "%s_output.txt", argv[0]);
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%.2f ", a_host[i]);
            }
            fprintf(fp, "\n");
            fclose(fp);
        }
    }
    free(a_host);
    return 0;
}
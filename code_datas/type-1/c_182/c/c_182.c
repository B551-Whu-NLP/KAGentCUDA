#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void Mul_half(float* src, float* dst, int size) {
    for (int i = 0; i < size; i++) {
        dst[i] = src[i] / 2.0f;
    }
}

int main(int argc, char *argv[]) {
    const int data_size = 100000000; // 增加100倍
    int verbose = 0;
    
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }
    
    float* src_host = (float*)malloc(data_size * sizeof(float));
    float* dst_host = (float*)malloc(data_size * sizeof(float));
    
    srand(42);
    for (int i = 0; i < data_size; ++i) {
        src_host[i] = (float)(rand() % 1000) / 10.0f;
    }

    clock_t start = clock();
    Mul_half(src_host, dst_host, data_size);
    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);

    if (verbose) {
        char output_file[256];
        snprintf(output_file, sizeof(output_file), "c_182_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%.2f ", dst_host[i]);
            }
            fprintf(fp, "\n");
            fclose(fp);
        }
    }

    free(src_host);
    free(dst_host);

    return 0;
}
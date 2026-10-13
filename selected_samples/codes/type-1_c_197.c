#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void const_fill(int N, float ALPHA, float* X, int INCX) {
    for (int i = 0; i < N; ++i) {
        X[i * INCX] = ALPHA;
    }
}

int main(int argc, char *argv[]) {
    const int data_size = 10000000; // 增大数据规模
    const float ALPHA = 8.0;
    float* X_host = (float*)malloc(data_size * sizeof(float));
    srand(42);
    for (int i = 0; i < data_size; ++i) {
        X_host[i] = (float)(rand() % 100) / 10.0;
    }
    int INCX = 3;

    clock_t start_time = clock();

    const_fill(data_size / INCX + (data_size % INCX ? 1 : 0), ALPHA, X_host, INCX);

    clock_t end_time = clock();

    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_197_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%.2f ", X_host[i]);
            }
            fprintf(fp, "\n");
            fclose(fp);
        }
    }

    free(X_host);
   
    printf("Time: %.2f(ms)\n", time_taken * 1000); return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void saxpy(float* x, float* y, float alpha, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = alpha * x[i] + y[i];
    }
}

int main(int argc, char* argv[]) {
    const int array_size = 10000000;
    float alpha = 0.5;
    float* x_host = (float*)malloc(array_size * sizeof(float));
    float* y_host = (float*)malloc(array_size * sizeof(float));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < array_size; ++i) {
        x_host[i] = i + 1.0;
        y_host[i] = (i + 1.0) * 2.0;
    }

    clock_t start = clock();

    saxpy(x_host, y_host, alpha, array_size);

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_213_output.txt");
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%.2f ", y_host[i]);
            }
            fclose(fp);
        }
    }

    free(x_host);
   
    printf("Time: %.2f(ms)\n", time_taken * 1000); free(y_host);

    return 0;
}
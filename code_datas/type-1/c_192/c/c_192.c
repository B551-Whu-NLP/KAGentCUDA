#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void vector_add(float *a, float *b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char *argv[]) {
    const int data_size = 10000000; // 增大数组大小
    int verbose = 0;
    
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    float* a_host = (float*)malloc(data_size * sizeof(float));
    float* b_host = (float*)malloc(data_size * sizeof(float));
    float* c_host = (float*)malloc(data_size * sizeof(float));
    
    srand(42);
    for (int i = 0; i < data_size; ++i) {
        a_host[i] = rand() % 1000;
        b_host[i] = rand() % 1000;
    }

    clock_t start = clock();
    
    vector_add(a_host, b_host, c_host, data_size);
    
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    if (verbose) {
        char output_file[256];
        snprintf(output_file, sizeof(output_file), "%s_output.txt", "c_192");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%.2f ", c_host[i]);
            }
            fprintf(fp, "\n");
            fclose(fp);
        }
    }

    free(a_host);
   
    printf("Time: %.2f(ms)\n", time_taken * 1000); free(b_host);
    free(c_host);

    return 0;
}
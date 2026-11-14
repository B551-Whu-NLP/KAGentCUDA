#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void update_x(double* x, double* a, double* b, int n) {
    for (int i = 0; i < n; ++i) {
        x[i] = a[i] * b[i];
    }
}

int main(int argc, char *argv[]) {
    int n = 1000000; // 增大数组大小
    double* h_x = (double*)malloc(n * sizeof(double));
    double* h_a = (double*)malloc(n * sizeof(double));
    double* h_b = (double*)malloc(n * sizeof(double));
    
    for (int i = 0; i < n; ++i) {
        h_x[i] = i + 1.0;
        h_a[i] = 2.0 * i + 1.0;
        h_b[i] = i + 2.0;
    }
    
    clock_t start = clock();
    
    // 增加迭代次数以延长执行时间
    for (int iter = 0; iter < 1000; ++iter) {
        update_x(h_x, h_a, h_b, n);
    }
    
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }
    
    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_264_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%.2f ", h_x[i]);
            }
            fclose(fp);
        }
    }
    
    free(h_x);
    free(h_a);
    free(h_b);
    
    printf("Time: %.2f(ms)\n", time_taken * 1000);
    
    return 0;
}
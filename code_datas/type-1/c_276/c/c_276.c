#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void kernel(double* x, int n) {
    for (int i = 0; i < n; i++) {
        double sum = 0.0;

        int limit = (i < 100) ? i : 100;
        for (int j = 1; j < 1000; j++) {
            double term = sqrt(pow(3.14159, limit)) / j;
            sum += term;
        }

        x[i] = sum;
    }
}

int main(int argc, char *argv[]) {
    int size = 1000;
    double* h_x = (double*)malloc(size * sizeof(double));
    
    srand(42);
    for (int i = 0; i < size; i++) {
        h_x[i] = (double)rand() / RAND_MAX;
    }

    clock_t start = clock();

    kernel(h_x, size);

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_276_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {    
                fprintf(fp, "%.10f ", h_x[i]);
            }
            fclose(fp);
        }
    } 

    free(h_x);

    printf("Time: %.2f(ms)\n", time_taken);
    return 0;
}
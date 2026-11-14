#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void gather_points(int b, int c, int n, int m, const float* points, const int* idx, float* out) {
    for (int i = 0; i < b; ++i) {
        for (int l = 0; l < c; ++l) {
            for (int j = 0; j < m; ++j) {
                int a = idx[i * m + j];
                out[(i * c + l) * m + j] = points[(i * c + l) * n + a];
            }
        }
    }
}

int main(int argc, char** argv) {
    int b = 1000; 
    int c = 30;  
    int n = 5000;
    int m = 100; 

    float* h_points = (float*)malloc(b * c * n * sizeof(float));
    int* h_idx = (int*)malloc(b * m * sizeof(int));
    float* h_out = (float*)malloc(b * c * m * sizeof(float));

    srand(0);
    for (int i = 0; i < b * c * n; ++i) {
        h_points[i] = (float)(rand() % 100);
    }
    for (int i = 0; i < b * m; ++i) {
        h_idx[i] = rand() % n; 
    }

    clock_t start = clock();
    gather_points(b, c, n, m, h_points, h_idx, h_out);
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", time_taken);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[] = "c_124_output.txt";
        FILE* file = fopen(output_file, "w");
        if (file != NULL) {
            for (int i = 0; i < 5; ++i) { 
                for (int j = 0; j < c; ++j) {
                    for (int k = 0; k < m; ++k) {
                        fprintf(file, "h_out[%d * %d + %d] = %f\n", i, c, k, h_out[(i * c + j) * m + k]);
                    }
                }
            }
            fclose(file);
        }
    }

    free(h_points);
    free(h_idx);
    free(h_out);

    return 0;
}
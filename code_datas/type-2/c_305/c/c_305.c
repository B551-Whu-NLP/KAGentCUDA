#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void manage_adj_matrix(float* graph, int n) {
    clock_t start = clock();

    for (int id = 0; id < n; ++id) {
        float sum = 0.0;
        for (int i = 0; i < n; ++i) {
            sum += graph[i * n + id];
        }
        for (int i = 0; i < n; ++i) {
            if (sum != 0.0) {
                graph[i * n + id] /= sum;
            }
            else {
                graph[i * n + id] = 1.0 / (float)n;
            }
        }
    }

    clock_t end = clock();
    double time_spent = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2fms\n", time_spent * 1000);
}

int main(int argc, char *argv[]) {
    int n = 1000;
    float *graph = (float *)malloc(n * n * sizeof(float));
    
    srand(42);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            graph[i * n + j] = rand() % 100 + 1.0;
        }
    }

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    manage_adj_matrix(graph, n);

    if (verbose) {
        FILE *fp = fopen("c_305_output.txt", "w");
        if (fp != NULL) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    fprintf(fp, "%.2f ", graph[i * n + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(graph);
    return 0;
}
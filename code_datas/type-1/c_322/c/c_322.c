#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int n = 1000;
    int *src2tgt = (int *)malloc(n * sizeof(int));
    float *adj = (float *)malloc(n * n * sizeof(float));
    float *prices = (float *)malloc(n * sizeof(float));    
    float *values = (float *)malloc(n * n * sizeof(float));
    float *bids = (float *)malloc(n * n * sizeof(float));
    unsigned short *complete = (unsigned short *)malloc(sizeof(unsigned short));
    
    srand(42);
    for (int i = 0; i < n; i++) {
        src2tgt[i] = -1;
        prices[i] = 0.5;

        for (int j = 0; j < n; j++) {
            adj[i * n + j] = (float)(rand() % 100) / 10.0f;
        }
    }

    clock_t start = clock();
    
    for (int idx = 0; idx < n * n; idx++) {
        int i = idx / n;
        int j = idx - i * n;
        bids[i * n + j] = -1;
        if (src2tgt[i] != -1) {
            continue;
        }
        complete[0] = 0;
        values[i * n + j] = -adj[i * n + j] - prices[j];
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2fms\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_322_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    fprintf(fp, "%.2f ", values[i * n + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(src2tgt);
    free(adj);
    free(prices);
    free(values);
    free(bids);
    free(complete);

    return 0;
}
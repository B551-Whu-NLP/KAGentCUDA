#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void update_clusters(int n, int k, double* Cx, double* Cy, double* Cx_sum, double* Cy_sum, int* Csize);

int main(int argc, char *argv[]) {
    const int k = 10000;
    const int n = 1000000;
    double* h_Cx = (double*)malloc(k * sizeof(double));
    double* h_Cy = (double*)malloc(k * sizeof(double));
    double* h_Cx_sum = (double*)malloc(k * sizeof(double));
    double* h_Cy_sum = (double*)malloc(k * sizeof(double));
    int* h_Csize = (int*)malloc(k * sizeof(int));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < k; ++i) {
        h_Cx[i] = i;
        h_Cy[i] = i;
        h_Cx_sum[i] = i;
        h_Cy_sum[i] = i;
        h_Csize[i] = i + 1;
    }

    clock_t start = clock();

    update_clusters(n, k, h_Cx, h_Cy, h_Cx_sum, h_Cy_sum, h_Csize);

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    if (verbose) {
        FILE *fp = fopen("c_290_output.txt", "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "Cluster %d: Cx=%.2f, Cy=%.2f\n", i, h_Cx[i], h_Cy[i]);
            }
            fclose(fp);
        }
    }

    free(h_Cx);
    free(h_Cy);
    free(h_Cx_sum);
    free(h_Cy_sum);
    free(h_Csize);

    printf("Time: %.2f(ms)\n", time_taken);
    return 0;
}

void update_clusters(int n, int k, double* Cx, double* Cy, double* Cx_sum, double* Cy_sum, int* Csize) {
    for (int iter = 0; iter < 10000; iter++) {
        for (int index = 0; index < k; index++) {
            if (Csize[index]) {
                Cx[index] = Cx_sum[index] / Csize[index];
                Cy[index] = Cy_sum[index] / Csize[index];
            }
        }
    }
}
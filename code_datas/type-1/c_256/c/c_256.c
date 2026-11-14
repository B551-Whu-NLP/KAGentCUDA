#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void zero_centroid_vals(int k, double* Cx_sum, double* Cy_sum, int* Csize) {
    for (int index = 0; index < k; index++) {
        Cx_sum[index] = 0;
        Cy_sum[index] = 0;
        Csize[index] = 0;
    }
}

int main(int argc, char *argv[]) {
    int k = 100000000; // 进一步增大k以显著增加计算量
    double* h_Cx_sum = (double*)malloc(k * sizeof(double));
    double* h_Cy_sum = (double*)malloc(k * sizeof(double));
    int* h_Csize = (int*)malloc(k * sizeof(int));

    clock_t start_time = clock();

    zero_centroid_vals(k, h_Cx_sum, h_Cy_sum, h_Csize);

    clock_t end_time = clock();
    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    printf("Time: %.2fms\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_256_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "Centroid %d: Cx_sum=%f, Cy_sum=%f, Csize=%d\n", i, h_Cx_sum[i], h_Cy_sum[i], h_Csize[i]);
            }
            fclose(fp);
        }
    }

    free(h_Cx_sum);
    free(h_Cy_sum);
    free(h_Csize);

    return 0;
}
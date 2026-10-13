#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

int main(int argc, char *argv[]) {
    const int size_x = 100000000;
    float* h_x_average = (float*)malloc(size_x * sizeof(float));
    float* h_x_outer_prod = (float*)malloc(size_x * sizeof(float));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < size_x; ++i) {
        h_x_average[i] = (float)(i);
    }

    clock_t start_time = clock();
    for (int i = 0; i < size_x; i++) {
        h_x_outer_prod[i] = h_x_average[i] * h_x_average[i];
    }
    clock_t end_time = clock();

    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", elapsed_time);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_65_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "h_x_outer_prod[%d]: %f\n", i, h_x_outer_prod[i]);
            }
            fclose(fp);
        }
    }

    free(h_x_average);
    free(h_x_outer_prod);

    return 0;
}
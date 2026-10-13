#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    const int arraySize = 100000000; // 增大数组大小
    float u_m = 5.0f;
    float u_d = 2.0f;
    float* h_u = (float*)malloc(arraySize * sizeof(float));
    float* h_lu = (float*)malloc(arraySize * sizeof(float));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < arraySize; ++i) {
        h_u[i] = (float)(i);
    }

    clock_t start = clock();

    for (int i = 0; i < arraySize; ++i) {
        h_lu[i] = (h_u[i] - u_m) / u_d;
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_43_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%.6f ", h_lu[i]);
            }
            fclose(fp);
        }
    }

    free(h_u);
    free(h_lu);
    return 0;
}
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

void memcpy_kernel(int* dst, int* src, size_t size) {
    memcpy(dst, src, size);
}

int main(int argc, char *argv[]) {
    int n = 10000000;
    int* h_src;
    int* h_dst;
    h_src = (int*)malloc(n * sizeof(int));
    h_dst = (int*)malloc(n * sizeof(int));

    for (int i = 0; i < n; ++i) {
        h_src[i] = i;
    }

    clock_t start = clock();

    memcpy_kernel(h_dst, h_src, n * sizeof(int));

    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_232_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%d ", h_dst[i]);
            }
            fclose(fp);
        }
    }
    free(h_src);
    free(h_dst);

    printf("Time: %.2f\n", time_taken);

    return 0;
}
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void setSuppressed(int* suppressed_host, int data_size) {
    for (int i = 0; i < data_size; i++) {
        suppressed_host[i] = i % 256;
    }
}

int main(int argc, char *argv[]) {
    const int data_size = 10000000;
    int* suppressed_host = (int*)malloc(data_size * sizeof(int));
    clock_t start = clock();
    setSuppressed(suppressed_host, data_size);
    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }
    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_185_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%d ", suppressed_host[i]);
            }
            fprintf(fp, "\n");
            fclose(fp);
        }
    }
    free(suppressed_host);
    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void setLabels(int* output, int dims, int clsNum) {
    for (int tid = 0; tid < dims; tid++) {
        output[tid] = tid % clsNum;
    }
}

int main(int argc, char *argv[]) {
    const int data_size = 10000000;
    const int clsNum = 3;
    int* output_host = (int*)malloc(data_size * sizeof(int));
    int verbose = 0;
    
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    clock_t start = clock();

    setLabels(output_host, data_size, clsNum);

    clock_t end = clock();
    double time_spent = ((double)(end - start)) / CLOCKS_PER_SEC;

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_193_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%d ", output_host[i]);
            }
            fprintf(fp, "\n");
            fclose(fp);
        }
    }

    free(output_host);
    printf("Time: %.2f(ms)\n", time_spent * 1000);
    return 0;
}
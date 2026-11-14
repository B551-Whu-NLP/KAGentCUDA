#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void setIndexYolov3(int* input, int dims, int batchSize) {
    for (int tid = 0; tid < dims; tid++) {
        for (int i = 0; i < batchSize; i++) {
            input[i * dims + tid] = tid;
        }
    }
}

int main(int argc, char *argv[]) {
    int dims = 10000;
    int batchSize = 5000;
    int* h_input = (int*)malloc(batchSize * dims * sizeof(int));
    
    srand(42);
    for (int i = 0; i < batchSize * dims; i++) {
        h_input[i] = rand() % 100;
    }

    clock_t start = clock();

    setIndexYolov3(h_input, dims, batchSize);

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        sprintf(output_filename, "%s_output.txt", argv[0]);
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                for (int j = 0; j < 100; ++j) {
                    fprintf(fp, "%d ", h_input[i * dims + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(h_input);
    printf("Time: %.2f(ms)\n", time_taken * 1000);
    return 0;
}
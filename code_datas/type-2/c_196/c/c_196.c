#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void kernelUpdateHead(int* head, int* d_idxs_out, int n) {
    for (int i = 0; i < n; i++) {
        head[d_idxs_out[i]] = 1;
    }
}

int main(int argc, char** argv) {
    const int data_size = 10000000;
    int* head_host = (int*)malloc(data_size * sizeof(int));
    int* d_idxs_out_host = (int*)malloc(data_size / 2 * sizeof(int));
    for (int i = 0; i < data_size; i++) {
        head_host[i] = 0;
    }
    for (int i = 0; i < data_size / 2; i++) {
        d_idxs_out_host[i] = 2 * i + 1;
    }

    clock_t start = clock();

    kernelUpdateHead(head_host, d_idxs_out_host, data_size / 2);

    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_196_output.txt");
        FILE* file = fopen(output_file, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%d ", head_host[i]);
            }
            fprintf(file, "\n");
            fclose(file);
        }
    }

    free(head_host);
    free(d_idxs_out_host);

    return 0;
}
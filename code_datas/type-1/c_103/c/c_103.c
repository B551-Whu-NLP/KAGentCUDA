#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void kernelMaximum(float* maxhd, float* maxvd, int start, int size) {
    int tx = start;
    float max_hd = 1.175494351e-38F;
    float max_vd = 1.175494351e-38F;

    clock_t start_time = clock();

    for (; tx < size; tx++) {
        if (maxhd[tx] > max_hd)
            max_hd = maxhd[tx];
        if (maxvd[tx] > max_vd)
            max_vd = maxvd[tx];
    }

    clock_t end_time = clock();
    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", elapsed_time);

    maxhd[0] = max_hd;
    maxvd[0] = max_vd;
}

int main(int argc, char** argv) {
    int start = 0;
    int size = 20480000; // 进一步增大数组大小
    float* h_maxhd = (float*)malloc(size * sizeof(float));
    float* h_maxvd = (float*)malloc(size * sizeof(float));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    srand(42);
    for (int i = 0; i < size; ++i) {
        h_maxhd[i] = (float)(rand() % 1000) / 10.0f;
    }
    for (int i = 0; i < size; ++i) {
        h_maxvd[i] = (float)(rand() % 1000) / 10.0f;
    }

    kernelMaximum(h_maxhd, h_maxvd, start, size);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_103_output.txt");
        FILE* f = fopen(output_file, "w");
        if (f != NULL) {
            fprintf(f, "Max_hd: %f\n", h_maxhd[0]);
            fprintf(f, "Max_vd: %f\n", h_maxvd[0]);
            fclose(f);
        }
    }

    free(h_maxhd);
    free(h_maxvd);

    return 0;
}
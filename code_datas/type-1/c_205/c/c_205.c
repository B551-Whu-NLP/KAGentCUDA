#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void vecAdd(float* in1, float* in2, float* out, int len) {
    for (int i = 0; i < len; i++) {
        out[i] = in1[i] + in2[i];
    }
}

int main(int argc, char *argv[]) {
    const int vector_size = 10000000;
    float* in1_host = (float*)malloc(vector_size * sizeof(float));
    float* in2_host = (float*)malloc(vector_size * sizeof(float));
    float* out_result = (float*)malloc(vector_size * sizeof(float));

    for (int i = 0; i < vector_size; ++i) {
        in1_host[i] = i + 1.0;
        in2_host[i] = (i + 1.0) * 2.0;
    }

    clock_t start = clock();
    vecAdd(in1_host, in2_host, out_result, vector_size);
    clock_t end = clock();

    double time_taken = ((double)(end - start) * 1000) / CLOCKS_PER_SEC;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_205_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%.2f ", out_result[i]);
            }
            fclose(fp);
        }
    }

    free(in1_host);
    free(in2_host);
    free(out_result);

    printf("Time: %.2f(ms)\n", time_taken);

    return 0;
}
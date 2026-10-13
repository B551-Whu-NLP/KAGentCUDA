#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void kComputeActs(const float *d_nets, float *d_acts, int size) {
    for (int un_idx = 0; un_idx < size; un_idx++) {
        float tact = 1.0f / (1.0f + expf(-d_nets[un_idx]));
        d_acts[un_idx] = tact;
    }
}

int main(int argc, char *argv[]) {
    const int N = 5000;
    float* h_nets = (float*)malloc(N * sizeof(float));
    float* h_acts = (float*)malloc(N * sizeof(float));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < N; ++i) {
        h_nets[i] = (float)(i);
        h_acts[i] = (float)(i);
    }

    clock_t start = clock();
    
    kComputeActs(h_nets, h_acts, N);
    
    clock_t end = clock();
    
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Time: %.2f(ms)\n", time_taken * 1000);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_54_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "acts[%d]: %f\n", i, h_acts[i]);
            }
            fclose(fp);
        }
    }

    free(h_nets);
    free(h_acts);

    return 0;
}
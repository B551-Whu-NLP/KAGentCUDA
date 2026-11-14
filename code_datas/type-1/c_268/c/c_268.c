#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void get_conf_inds(const float* mlvl_conf, const float conf_thr, int* conf_inds, int dims) {
    for (int tid = 0; tid < dims; tid++) {
        if (mlvl_conf[tid] >= conf_thr) {
            conf_inds[tid] = 1;
        }
        else {
            conf_inds[tid] = -1;
        }
    }
}

int main(int argc, char* argv[]) {
    int dims = 10000000;
    float conf_thr = 0.5;
    float* h_mlvl_conf = (float*)malloc(dims * sizeof(float));
    int* h_conf_inds = (int*)malloc(dims * sizeof(int));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    srand(42);
    for (int i = 0; i < dims; ++i) {
        h_mlvl_conf[i] = rand() / (float)RAND_MAX;
    }

    clock_t start = clock();
    get_conf_inds(h_mlvl_conf, conf_thr, h_conf_inds, dims);
    clock_t end = clock();

    double time_spent = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    if (verbose) {
        char output_file[256];
        sprintf(output_file, "%s_output.txt", argv[0]);
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%.2f ", h_mlvl_conf[i]);
            }
            fprintf(fp, "\n");
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%d ", h_conf_inds[i]);
            }
            fclose(fp);
        }
    }

    free(h_mlvl_conf);
    free(h_conf_inds);

    printf("Time: %.2f(ms)\n", time_spent);

    return 0;
}
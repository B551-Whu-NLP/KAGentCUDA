#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void compute_new_means(float *mx, float *my, const float *sx, const float *sy, const int *c) {
    for (int cluster = 0; cluster < 5; cluster++) {
        const int count = (c[cluster] > 0) ? c[cluster] : 1;
        mx[cluster] = sx[cluster] / count;
        my[cluster] = sy[cluster] / count;
    }
}

int main(int argc, char *argv[]) {
    const int clusters = 5;
    const int iterations = 100000000; // 显著增加迭代次数以延长执行时间

    float* h_mx = (float*)malloc(clusters * sizeof(float));
    float* h_my = (float*)malloc(clusters * sizeof(float));
    float* h_sx = (float*)malloc(clusters * sizeof(float));
    float* h_sy = (float*)malloc(clusters * sizeof(float));
    int* h_c = (int*)malloc(clusters * sizeof(int));

    for (int i = 0; i < clusters; ++i) {
        h_sx[i] = i + 1;
        h_sy[i] = i + 1;
        h_c[i] = i + 1;
    }
    
    clock_t start = clock();

    for (int iter = 0; iter < iterations; iter++) {
        compute_new_means(h_mx, h_my, h_sx, h_sy, h_c);
    }

    clock_t end = clock();

    double time_spent = ((double)(end - start)) / CLOCKS_PER_SEC;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_73_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "(%f, %f)\n", h_mx[i], h_my[i]);
            }
            fclose(fp);
        }
    }

    free(h_mx);
    free(h_my);
    free(h_sx);
    free(h_sy);
    free(h_c);

    printf("Time: %.2f(ms)\n", time_spent * 1000);

    return 0;
}
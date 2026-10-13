#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void saxpy(float *vecY, float *vecX, float alpha, int n) {
    for (int i = 0; i < n; i++) {
        vecY[i] = alpha * vecX[i] + vecY[i];
    }
}

int main(int argc, char *argv[]) {
    const int n = 1024 * 2048 * 64;
    int verbose = 0;
    
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    float* h_vecY = (float*)malloc(n * sizeof(float));
    float* h_vecX = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n; ++i) {
        h_vecY[i] = (float)(i);
        h_vecX[i] = (float)(i * 2);
    }

    clock_t start = clock();

    saxpy(h_vecY, h_vecX, 2.0f, n);

    clock_t end = clock();

    double time_spent = ((double)(end - start) / CLOCKS_PER_SEC) * 1000;

    if (verbose) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "%s_output.txt", "c_66");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "h_vecY[%d]: %f\n", i, h_vecY[i]);
            }
            fclose(fp);
        }
    }

    free(h_vecY);
    free(h_vecX);

    printf("Time: %.2f(ms)\n", time_spent);
    return 0;
}
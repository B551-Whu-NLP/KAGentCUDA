#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void invalidateFlow(float* modFlowX, float* modFlowY, const float* constFlowX,
    const float* constFlowY, int width, int height, float cons_thres, int verbose) {
    clock_t start = clock();

    for (int iter = 0; iter < 1000; iter++) {
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                int ind = y * width + x;
                float mFX = modFlowX[ind];
                float mFY = modFlowY[ind];
                float cFX = constFlowX[ind];
                float cFY = constFlowY[ind];
                float err = (mFX - cFX) * (mFX - cFX) + (mFY - cFY) * (mFY - cFY);
                if (err > cons_thres) {
                    mFX = 0;
                    mFY = 0;
                }
                modFlowX[ind] = mFX;
                modFlowY[ind] = mFY;
            }
        }
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);

    if (verbose) {
        char output_file[256];
        snprintf(output_file, sizeof(output_file), "c_340_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "modFlowX[%d] = %.2f, modFlowY[%d] = %.2f\n", i, modFlowX[i], i, modFlowY[i]);
            }
            fclose(fp);
        }
    }
}

int main(int argc, char *argv[]) {
    const int width = 1024;
    const int height = 768;
    const float cons_thres = 0.1;
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    float* modFlowX, * modFlowY, * constFlowX, * constFlowY;

    size_t flow_size = width * height * sizeof(float);

    modFlowX = (float*)malloc(flow_size);
    modFlowY = (float*)malloc(flow_size);
    constFlowX = (float*)malloc(flow_size);
    constFlowY = (float*)malloc(flow_size);

    srand(0);
    for (int i = 0; i < width * height; ++i) {
        modFlowX[i] = (float)(rand() % 1000) / 100.0f;
        modFlowY[i] = (float)(rand() % 1000) / 100.0f;
        constFlowX[i] = (float)(rand() % 1000) / 100.0f;
        constFlowY[i] = (float)(rand() % 1000) / 100.0f;
    }

    invalidateFlow(modFlowX, modFlowY, constFlowX, constFlowY, width, height, cons_thres, verbose);

    free(modFlowX);
    free(modFlowY);
    free(constFlowX);
    free(constFlowY);
    return 0;
}
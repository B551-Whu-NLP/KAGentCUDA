#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void LreluBackward(float* srcDiff, float* dstDiff, float* srcData, int data_size, float alpha) {
    for (int i = 0; i < data_size; i++) {
        dstDiff[i] = (srcData[i] > 0) ? srcDiff[i] * 1.0 : srcDiff[i] * alpha;
    }
}

int main(int argc, char** argv) {
    int data_size = 10240000; // 进一步增大数据规模
    float alpha = 0.01;
    float* h_srcDiff = (float*)malloc(data_size * sizeof(float));
    float* h_dstDiff = (float*)malloc(data_size * sizeof(float));
    float* h_srcData = (float*)malloc(data_size * sizeof(float));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < data_size; ++i) {
        h_srcDiff[i] = 1.0f;
        h_srcData[i] = (float)(i % 256) - 128.0f;
    }

    clock_t start = clock();

    // 多次调用以增加总执行时间
    for (int iter = 0; iter < 1000; iter++) { // 增加迭代次数
        LreluBackward(h_srcDiff, h_dstDiff, h_srcData, data_size, alpha);
    }

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", time_taken);

    if (verbose) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "%s_output.txt", "c_86");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "%d: srcData = %f, srcDiff = %f, dstDiff = %f\n", i, h_srcData[i], h_srcDiff[i], h_dstDiff[i]);
            }
            fclose(file);
        }
    }

    free(h_srcDiff);
    free(h_dstDiff);
    free(h_srcData);

    return 0;
}
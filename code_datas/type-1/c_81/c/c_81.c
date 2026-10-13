#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void LreluForward(float *srcData, float *dstData, int data_size, float alpha) {
    for (int i = 0; i < data_size; i++) {
        dstData[i] = srcData[i] > 0 ? srcData[i] : srcData[i] * alpha;
    }
}

int main(int argc, char *argv[]) {
    int data_size = 1024000; // 增大数据规模
    float alpha = 0.1f;
    float* h_srcData = (float*)malloc(data_size * sizeof(float));
    float* h_dstData = (float*)malloc(data_size * sizeof(float));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    srand(12345);
    for (int i = 0; i < data_size; ++i) {
        h_srcData[i] = (float)(rand() % 100 - 50);
    }

    clock_t start = clock();

    // 增加迭代次数以延长执行时间
    for (int iter = 0; iter < 100; iter++) {
        LreluForward(h_srcData, h_dstData, data_size, alpha);
    }

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_81_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%.2f ", h_dstData[i]);
            }
            fclose(fp);
        }
    }

    free(h_srcData);
    printf("Time: %.2fms\n", time_taken * 1000);
    free(h_dstData);

    return 0;
}
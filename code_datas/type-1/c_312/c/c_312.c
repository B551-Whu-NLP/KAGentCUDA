#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void SetToZero_kernel(float* vx, float* vy, float* vz, int w, int h, int l) {
    unsigned int i, j;

    for (i = 0; i < w; i++) {
        for (j = 0; j < h; j++) {
            unsigned int index = j * w + i;
            for (int k = 0; k < l; ++k, index += w * h) {
                vx[index] = 0;
                vy[index] = 0;
                vz[index] = 0;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int w = 500; 
    int h = 500;
    int l = 500;

    float *vx = (float *)malloc(w * h * l * sizeof(float));
    float *vy = (float *)malloc(w * h * l * sizeof(float));
    float *vz = (float *)malloc(w * h * l * sizeof(float));

    // 初始化数据
    for (int i = 0; i < w * h * l; i++) {
        vx[i] = 1.0f;
        vy[i] = 1.0f;
        vz[i] = 1.0f;
    }

    // 记录开始时间
    clock_t start = clock();

    SetToZero_kernel(vx, vy, vz, w, h, l);

    // 记录结束时间
    clock_t end = clock();

    // 计算运行时间（单位：毫秒）
    double time_spent = ((double)(end - start) / CLOCKS_PER_SEC) * 1000;

    // 打印运行时间
    printf("Time: %.2f(ms)\n", time_spent);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_312_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%.2f ", vx[i]);
            }
            fprintf(fp, "\n");
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%.2f ", vy[i]);
            }
            fprintf(fp, "\n");
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%.2f ", vz[i]);
            }
            fprintf(fp, "\n");
            fclose(fp);
        }
    }

    free(vx);
    free(vy);
    free(vz);

    return 0;
}
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void Blend(unsigned char *aR1, unsigned char *aR2, unsigned char *aRS, int size) {
    for (int i = 0; i < size; ++i) {
        aRS[i] = (unsigned char)(0.5 * aR1[i] + 0.5 * aR2[i]);
    }
}

int main(int argc, char *argv[]) {
    const int size = 10000000; // 增大数组大小
    unsigned char* h_aR1 = (unsigned char*)malloc(size * sizeof(unsigned char));
    unsigned char* h_aR2 = (unsigned char*)malloc(size * sizeof(unsigned char));
    unsigned char* h_aRS = (unsigned char*)malloc(size * sizeof(unsigned char));

    for (int i = 0; i < size; ++i) {
        h_aR1[i] = (unsigned char)(i * 2); 
        h_aR2[i] = (unsigned char)(i * 3);
    }

    clock_t start = clock();

    // 多次调用Blending函数以增加总计算量
    for (int iter = 0; iter < 100; ++iter) {
        Blend(h_aR1, h_aR2, h_aRS, size);
    }

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", time_taken);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_62_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%u\n", h_aRS[i]);
            }
            fclose(fp);
        }
    }

    free(h_aR1);
    free(h_aR2);
    free(h_aRS);

    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void sumAndScale(float* noiseVariance, float* diffMag2, int n, int verbose, const char* file_name) {
    clock_t start = clock();

    for (int i = 0; i < n; i++) {
        int batchJump = i * 347;
        float temp = 0;
        for (int sumIndex = 0; sumIndex < 347; sumIndex++) {
            temp += diffMag2[batchJump + sumIndex];
        }
        temp = 0.00161812 * temp;
        noiseVariance[i] = temp;
    }

    clock_t end = clock();
    double timeSpent = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Time: %.2f(ms)\n", timeSpent);

    if (verbose) {
        char output_file[256];
        sprintf(output_file, "%s_output.txt", file_name);
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "Element %d: %f\n", i, noiseVariance[i]);
            }
            fclose(fp);
        }
    }
}

int main(int argc, char* argv[]) {
    const int n = 100000; // 增大数组大小
    float* h_diffMag2 = (float*)malloc(n * 347 * sizeof(float));
    float* h_noiseVariance = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n * 347; ++i) {
        h_diffMag2[i] = (float)(i % 100);
    }

    int verbose = 0;
    const char* file_name = "c_293";

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    sumAndScale(h_noiseVariance, h_diffMag2, n, verbose, file_name);


    free(h_diffMag2);
    free(h_noiseVariance);

    return 0;
}
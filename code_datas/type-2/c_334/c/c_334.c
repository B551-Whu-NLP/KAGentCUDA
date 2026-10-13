#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void permuteData2(const float* input, float* output, int num, int devideNum, int featureSize, int priorNum, int batchSize) {
    clock_t start = clock();

    for (int tid = 0; tid < num; tid++) {
        int numPerbatch = num * devideNum * priorNum;
        for (int s = 0; s < batchSize; s++) {
            for (int i = 0; i < priorNum; i++) {
                for (int j = 0; j < devideNum; j++) {
                    output[s * numPerbatch + tid * priorNum * devideNum + i * devideNum + j] =
                        input[s * numPerbatch + (i * devideNum * featureSize) + (j * featureSize) + tid];
                }
            }
        }
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);
}

int main(int argc, char *argv[]) {
    const int num = 128;
    const int devideNum = 64;
    const int featureSize = 128;
    const int priorNum = 32;
    const int batchSize = 4;

    float* input_host = (float*)malloc(num * devideNum * priorNum * featureSize * batchSize * sizeof(float));
    float* output_host = (float*)malloc(num * devideNum * priorNum * batchSize * sizeof(float));

    if (input_host == NULL || output_host == NULL) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    for (int s = 0; s < batchSize; s++) {
        for (int i = 0; i < priorNum; i++) {
            for (int j = 0; j < devideNum; j++) {
                for (int k = 0; k < num; k++) {
                    input_host[s * num * devideNum * priorNum * featureSize + i * devideNum * featureSize + j * featureSize + k] =
                        (float)(s * 1000 + i * 100 + j * 10 + k);
                }
            }
        }
    }

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    permuteData2(input_host, output_host, num, devideNum, featureSize, priorNum, batchSize);

    if (verbose) {
        FILE *fp = fopen("c_334_output.txt", "w");
        if (fp != NULL) {
            int output_size = batchSize * num * priorNum * devideNum;
            int print_count = (output_size < 1000) ? output_size : 1000;
            for (int i = 0; i < print_count; i++) {
                fprintf(fp, "%.2f ", output_host[i]);
                if ((i + 1) % 16 == 0) {
                    fprintf(fp, "\n");
                }
            }
            fprintf(fp, "\n");
            fclose(fp);
        }
    }

    free(input_host);
    free(output_host);
    return 0;
}

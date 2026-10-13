#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void distanceMatFinal(long int totalPixels, int availablePixels, int outPixelOffset, float* distMat) {
    clock_t start = clock();

    for (long int i = 0; i < availablePixels; i++) {
        float sum = 0.0;
        float max = 0.0;

        for (long int j = 0; j < totalPixels; j++) {
            float element = distMat[i * totalPixels + j];
            if (element > max) max = element;
            sum += element;
        }

        sum += max;

        for (long int j = 0; j < totalPixels; j++) {
            if ((i + outPixelOffset) == j)
                distMat[i * totalPixels + j] = max / sum;
            else
                distMat[i * totalPixels + j] /= sum;
        }
    }

    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_spent * 1000);
}

int main(int argc, char *argv[]) {
    const long int totalPixels = 1000;
    const int availablePixels = 500;
    const int outPixelOffset = 10;
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    float* distMat_host = (float*)malloc(totalPixels * availablePixels * sizeof(float));

    srand(42);
    for (long int i = 0; i < availablePixels; i++) {
        for (long int j = 0; j < totalPixels; j++) {
            distMat_host[i * totalPixels + j] = (float)(rand() % 100) / 100.0f;
        }
    }

    distanceMatFinal(totalPixels, availablePixels, outPixelOffset, distMat_host);

    if (verbose) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "c_333_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < 100; i++) {
                for (int j = 0; j < 50; j++) {
                    fprintf(fp, "%f ", distMat_host[i * totalPixels + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(distMat_host);
    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void vectorMatrixMult(long int totalPixels, float* matrix, float* vector, float* out) {
    for (long int i = 0; i < totalPixels; i++) {
        float sum = 0.0;
        for (long int j = 0; j < totalPixels; j++) {
            sum += matrix[i * totalPixels + j] * vector[j];
        }
        out[i] = sum;
    }
}

int main(int argc, char *argv[]) {
    const long int totalPixels = 10000;
    float* h_matrix = (float*)malloc(totalPixels * totalPixels * sizeof(float));
    float* h_vector = (float*)malloc(totalPixels * sizeof(float));
    float* h_out = (float*)malloc(totalPixels * sizeof(float));

    if (!h_matrix || !h_vector || !h_out) {
        fprintf(stderr, "Memory allocation failed.\n");
        return 1;
    }

    for (long int i = 0; i < totalPixels * totalPixels; ++i) {
        h_matrix[i] = i % 100;
    }
    for (long int i = 0; i < totalPixels; ++i) {
        h_vector[i] = i % 100;
    }

    clock_t start = clock();
    vectorMatrixMult(totalPixels, h_matrix, h_vector, h_out);
    clock_t end = clock();

    double timeElapsed = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Time: %.2f(ms)\n", timeElapsed);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_288_output.txt");
        FILE *fp = fopen(outputFileName, "w");
        if (fp != NULL) {
            for (long int i = 0; i < 1000; ++i) {
                fprintf(fp, "%.2f\t", h_out[i]);
            }
            fclose(fp);
        }
    }

    free(h_matrix);
    free(h_vector);
    free(h_out);

    return 0;
}
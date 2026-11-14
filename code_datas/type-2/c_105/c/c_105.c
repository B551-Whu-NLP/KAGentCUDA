#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void vectorMatrixMult(long int totalPixels, int availablePixels, int outPixelOffset, float* matrix, float* vector, float* out, int verbose, const char* file_name) {
    clock_t start = clock();

    for (long int i = 0; i < availablePixels; i++) {
        float sum = 0.0;
        for (long int j = 0; j < totalPixels; j++) {
            sum += matrix[i * totalPixels + j] * vector[j];
        }
        out[i + outPixelOffset] = sum;
    }

    clock_t end = clock();
    double timeElapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2fms\n", timeElapsed * 1000);

    if (verbose) {
        char output_file[256];
        snprintf(output_file, sizeof(output_file), "%s_output.txt", file_name);
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%.6f ", out[i]);
            }
            fprintf(fp, "\n");
            fclose(fp);
        }
    }
}

int main(int argc, char** argv) {
    long int totalPixels = 10000;
    int availablePixels = 5000;
    int outPixelOffset = 2500;
    int verbose = 0;
    const char* file_name = "c_105";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        }
    }

    float* h_matrix = (float*)malloc(availablePixels * totalPixels * sizeof(float));
    float* h_vector = (float*)malloc(totalPixels * sizeof(float));
    float* h_out = (float*)malloc((availablePixels + outPixelOffset) * sizeof(float));

    srand(42);
    for (int i = 0; i < availablePixels * totalPixels; i++) {
        h_matrix[i] = (float)(rand() % 1000) / 100.0f;
    }
    for (int i = 0; i < totalPixels; i++) {
        h_vector[i] = (float)(rand() % 1000) / 100.0f;
    }

    vectorMatrixMult(totalPixels, availablePixels, outPixelOffset, h_matrix, h_vector, h_out, verbose, file_name);

    free(h_matrix);
    free(h_vector);
    free(h_out);
    return 0;
}
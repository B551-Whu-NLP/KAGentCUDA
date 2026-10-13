#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void bitPrune(unsigned char* out, float* in, int frontPrune, int outputlength, int inputLength, int n) {
    for (int i = 0; i < n; i++) {
        int batch = i / outputlength;
        int indexInBatch = i % outputlength;
        int batchInJump = batch * inputLength;
        int indexOutBatch = i % outputlength;
        int batchOutJump = batch * outputlength;
        int frontJump = frontPrune;
        out[batchOutJump + indexOutBatch] = (char)(in[batchInJump + frontJump + indexInBatch] > 0);
    }
}

int main(int argc, char** argv) {
    int frontPrune = 10;
    int outputLength = 10;
    int inputLength = 12;
    int n = 10000000; // Increased from 1000000 to 10000000
    unsigned char* h_out = (unsigned char*)malloc(n * outputLength * sizeof(unsigned char));
    float* h_in = (float*)malloc(n * inputLength * sizeof(float));

    for (int i = 0; i < n * inputLength; i++) {
        h_in[i] = (float)(i % 100);
    }
    for (int i = 0; i < n * outputLength; i++) {
        h_out[i] = (unsigned char)(0);
    }

    clock_t start = clock();
    bitPrune(h_out, h_in, frontPrune, outputLength, inputLength, n);
    clock_t end = clock();
    double timeTaken = ((double)(end - start) / CLOCKS_PER_SEC) * 1000;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_131_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%u ", h_out[i]);
            }
            fprintf(file, "\n");
            fclose(file);
        }
    }

    free(h_out);
    printf("Time: %.2f(ms)\n", timeTaken);
    free(h_in);
    return 0;
}
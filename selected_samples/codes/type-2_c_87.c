#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void Reduce(int* I, int* O, unsigned int n) {
    unsigned int blockSize = 256; 
    unsigned int numBlocks = (n + blockSize - 1) / blockSize; 
    unsigned int threadsPerBlock = blockSize;

    int* N = (int*)malloc(numBlocks * threadsPerBlock * sizeof(int));
    if (N == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    for (unsigned int i = 0; i < numBlocks; ++i) {
        for (unsigned int j = 0; j < threadsPerBlock; ++j) {
            unsigned int idx = i * threadsPerBlock + j;
            if (idx < n) {
                N[i * threadsPerBlock + j] = I[idx];
            }
            else {
                N[i * threadsPerBlock + j] = 0;
            }
        }
    }
    for (int stride = 1; stride < threadsPerBlock; stride *= 2) {
        for (unsigned int i = 0; i < numBlocks; ++i) {
            for (unsigned int tid = 0; tid < threadsPerBlock; ++tid) {
                if ((tid % (2 * stride)) == 0 && tid + stride < threadsPerBlock) {
                    N[i * threadsPerBlock + tid] += N[i * threadsPerBlock + tid + stride];
                }
            }
        }
    }
    for (unsigned int i = 0; i < numBlocks; ++i) {
        O[i] = N[i * threadsPerBlock];
    }
    free(N);
}

int main(int argc, char** argv) {
    unsigned int n = 10240;
    size_t bytes = n * sizeof(int);
    int* h_I = (int*)malloc(bytes);
    int* h_O = (int*)malloc(n * sizeof(int));

    for (unsigned int i = 0; i < n; ++i) {
        h_I[i] = i + 1;
    }

    clock_t start = clock();
    Reduce(h_I, h_O, n);
    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time: %.2fms\n", time_taken * 1000);

    int sum = 0;
    for (int i = 0; i < (n + 256 - 1) / 256; ++i) {
        sum += h_O[i];
    }

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_87_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            fprintf(file, "%d\n", sum);
            fclose(file);
        }
    }

    free(h_I);
    free(h_O);

    return 0;
}
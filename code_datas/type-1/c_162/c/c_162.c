#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void calculateOuterSums(float* innerSums, float* L, int uLength) {
    for (int u = 0; u < uLength; u++) {
        float real, imag, u_sum; 
        int realIdx = 2 * u; 
        int imagIdx = realIdx + 1; 
        real = innerSums[realIdx]; 
        imag = innerSums[imagIdx]; 
        u_sum = (real * real) + (imag * imag);
        realIdx += 64; 
        imagIdx += 64; 
        real = innerSums[realIdx]; 
        imag = innerSums[imagIdx];
        u_sum += (real * real) + (imag * imag); 
        realIdx += 64; imagIdx += 64;
        real = innerSums[realIdx]; 
        imag = innerSums[imagIdx]; 
        u_sum += (real * real) + (imag * imag); 
        realIdx += 64; imagIdx += 64; 
        real = innerSums[realIdx]; 
        imag = innerSums[imagIdx];
        u_sum += (real * real) + (imag * imag); 
        realIdx += 64; imagIdx += 64;
        real = innerSums[realIdx]; 
        imag = innerSums[imagIdx]; 
        u_sum += (real * real) + (imag * imag); 
        realIdx += 64; imagIdx += 64;
        real = innerSums[realIdx]; 
        imag = innerSums[imagIdx]; 
        u_sum += (real * real) + (imag * imag); 
        realIdx += 64; imagIdx += 64;
        real = innerSums[realIdx];
        imag = innerSums[imagIdx];
        u_sum += (real * real) + (imag * imag); 
        realIdx += 64; imagIdx += 64; 
        real = innerSums[realIdx];
        imag = innerSums[imagIdx];
        u_sum += (real * real) + (imag * imag);
        
        // Additional iterations to increase computation time
        for (int i = 0; i < 100000; i++) {
            u_sum += (real * real) + (imag * imag);
        }
        L[u] = u_sum;
    }
}

int main(int argc, char *argv[]) {
    int uLength = 1000000;  // Increased from 100000 to 1000000
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    float* innerSums_host = (float*)malloc(2 * uLength * sizeof(float));
    float* L_host = (float*)malloc(uLength * sizeof(float));

    for (int i = 0; i < 2 * uLength; i++) {
        innerSums_host[i] = (float)(i % 10);
    }
    calculateOuterSums(innerSums_host, L_host, uLength);

    if (verbose) {
        char outputFileName[256];
        snprintf(outputFileName, sizeof(outputFileName), "c_162_output.txt");
        FILE *fp = fopen(outputFileName, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "L[%d] = %f\n", i, L_host[i]);
            }
            fclose(fp);
        }
    }
    free(innerSums_host);
    free(L_host);
 
    return 0;
}
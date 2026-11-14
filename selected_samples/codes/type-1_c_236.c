#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int batchSize = 10000000;
    int dims = 100000;
    long long* h_offset = (long long*)malloc((batchSize + 1) * sizeof(long long));
    
    clock_t start = clock();
    
    h_offset[0] = 0;
    for (int i = 1; i <= batchSize; i++) {
        h_offset[i] = (long long)i * dims;
    }
    
    clock_t end = clock();
    double timeElapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("Time: %.2fms\n", timeElapsed * 1000);
    
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }
    
    if (verbose) {
        char outputFileName[] = "c_236_output.txt";
        FILE *fp = fopen(outputFileName, "w");
        if (fp != NULL) {
            for (int i = 0; i <= 1000; i++) {
                fprintf(fp, "%lld ", h_offset[i]);
            }
            fclose(fp);
        }
    }
    
    free(h_offset);
    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void testInt1(const int *input, int dims, int verbose) {
    clock_t start = clock();
    
    int *result = (int*)malloc(dims * sizeof(int));
    for (int tid = 0; tid < dims; tid++) {
        int sum = 0;
        for (int i = 0; i < 3000 * 4; i++) {
            if (input[i] == 0) {
                sum++;
            }
        }
        result[tid] = sum;
    }
    
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    
    printf("Time: %.2f(ms)\n", time_taken);
    
    if (verbose) {
        char output_file[256];
        snprintf(output_file, sizeof(output_file), "c_57_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%d ", result[i]);
            }
            fclose(fp);
        }
    }
    free(result);
}

int main(int argc, char *argv[]) {
    const int dims = 3000 * 4;
    int* h_input = (int*)malloc(dims * sizeof(int));
    
    for (int i = 0; i < dims; ++i) {
        h_input[i] = i % 2;
    }
    
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }
    
    testInt1(h_input, dims, verbose);
    free(h_input);
    return 0;
}

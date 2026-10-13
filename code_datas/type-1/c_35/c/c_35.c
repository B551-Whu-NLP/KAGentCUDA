#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void activate_array_leaky(float* x, int n) {
    for (int index = 0; index < n; index++) {
        float val = x[index];
        x[index] = (val > 0) ? val : val / 10;
    }
}

int main(int argc, char* argv[]) {
    const int arraySize = 1000000;
    float* h_x = (float*)malloc(arraySize * sizeof(float));
    int verbose = 0;
    
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    srand(42);
    for (int i = 0; i < arraySize; ++i) {
        h_x[i] = (float)(rand() % 1000 - 500);
    }

    clock_t start = clock();
    activate_array_leaky(h_x, arraySize);
    clock_t end = clock();
    
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_35_output.txt");
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            fprintf(fp, "Time: %.2f(ms)\n", time_taken);
            fprintf(fp, "激活后的数组：\n");
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%.2f ", h_x[i]);
            }
            fprintf(fp, "\n");
            fclose(fp);
        }
    }
    
    free(h_x);
    printf("Time: %.2f(ms)\n", time_taken);
    return 0;
}
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void core_computation(int *arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = i * 2 + 1;
    }
}

int main(int argc, char *argv[]) {
    clock_t start, end;
    double cpu_time_used;
    int size = 100000000;
    int *arr = (int*)malloc(size * sizeof(int));
    
    if (arr == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }
    
    start = clock();
    core_computation(arr, size);
    end = clock();
    
    cpu_time_used = ((double)(end - start)) * 1000.0 / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", cpu_time_used);
    
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }
    
    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_147_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 100; i++) {
                fprintf(fp, "%d ", arr[i]);
            }
            fprintf(fp, "\n");
            fclose(fp);
        }
    }
    
    free(arr);
    return 0;
}